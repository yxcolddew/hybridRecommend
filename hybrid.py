def main(folder_path, test_file_name, output_file_name, alpha=0.3):
    sc = SparkContext(appName='Hybrid_task2_3')
    sc.setLogLevel("ERROR")
    start_time = time.time()

    # Load and preprocess training and test data
    training_rdd = load_and_preprocess_rdd(sc, folder_path + '/yelp_train.csv', 30)
    test_rdd = load_and_preprocess_rdd(sc, test_file_name, 30)

    # CF Average Ratings
    user_avg_rating = compute_average_ratings(training_rdd, 0)
    business_avg_rating = compute_average_ratings(training_rdd, 1)

    # Collaborative Filtering Predictions
    cf_predictions = test_rdd.map(
        lambda x: (
            (x.split(",")[0], x.split(",")[1]),  
            predict_rating_cf(x.split(",")[0], x.split(",")[1], user_avg_rating, business_avg_rating) 
        )
    ).collectAsMap()

    # Load additional data for model-based approach
    user_features_rdd = sc.textFile(folder_path + '/user.json').map(lambda x: json.loads(x))
    business_features_rdd = sc.textFile(folder_path + '/business.json').map(lambda x: json.loads(x))

    # Create feature dictionaries
    user_features_dict = create_user_features_dict(user_features_rdd)
    business_features_dict = create_business_features_dict(business_features_rdd)
    
    # Extract features for model training and prediction
    train_data_rdd = training_rdd.map(lambda x: x.split(','))
    test_data_rdd = test_rdd.map(lambda x: x.split(','))

    train_x = extract_features(train_data_rdd, user_features_dict, business_features_dict)
    train_y = np.array([float(x[2]) for x in train_data_rdd.collect()])
    test_x = extract_features(test_data_rdd, user_features_dict, business_features_dict)

    # Train XGBoost model
    model = xgb.XGBRegressor(max_depth=10,
                             learning_rate=0.1,
                             n_estimators=100,
                             objective='reg:linear',
                             booster='gbtree',
                             gamma=0,
                             min_child_weight=1,
                             subsample=1,
                             colsample_bytree=1,
                             reg_alpha=0,
                             reg_lambda=1,
                             random_state=0)
    model.fit(train_x, train_y)

    # Model-based predictions
    model_based_predictions = model.predict(test_x)

    # CF RMSE Calculation
    actual_ratings = np.array(test_data_rdd.map(lambda x: float(x[2])).collect())
    cf_ratings = np.array([cf_predictions.get((x[0], x[1]), 2.5) for x in test_data_rdd.collect()])
    rmse_cf = compute_rmse(cf_ratings, actual_ratings)
    print(f"RMSE for Collaborative Filtering: {rmse_cf}")

    # Model-based RMSE Calculation
    rmse_model = compute_rmse(model_based_predictions, actual_ratings)
    print(f"RMSE for Model-Based Approach: {rmse_model}")

    # Combine predictions
    combined_predictions = []
    for idx, (user_business, cf_prediction) in enumerate(cf_predictions.items()):
        user_id, business_id = user_business[0], user_business[1]
        model_based_prediction = model_based_predictions[idx]
        final_prediction = alpha * cf_prediction + (1 - alpha) * model_based_prediction
        combined_predictions.append((user_id, business_id, final_prediction))

    # Compute Combined RMSE
    predicted_ratings = np.array([pred for _, _, pred in combined_predictions])
    rmse_combined = compute_rmse(predicted_ratings, actual_ratings)
    print(f"Combined RMSE: {rmse_combined}")

    # Save predictions to output file
    with open(output_file_name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "business_id", "prediction"])
        for user_id, business_id, prediction in combined_predictions:
            writer.writerow([user_id, business_id, prediction])

    # Duration output
    end_time = time.time()
    print("Duration: ", end_time - start_time)
    
    sc.stop()

if __name__ == "__main__":
    folder_path = sys.argv[1]
    test_file_name = sys.argv[2]
    output_file_name = sys.argv[3]
    main(folder_path, test_file_name, output_file_name)
