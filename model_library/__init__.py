import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error

def regression_models(X, y, max_depth=None, leaf_nodes=1, n_neighbors=5):
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    
    # Scale the features using StandardScaler
    scaler = StandardScaler()
    train_df = scaler.fit_transform(X_train)
    test_df = scaler.transform(X_test)
    
    # Convert the scaled arrays back to DataFrames
    X_train = pd.DataFrame(train_df, columns=X_train.columns)
    X_test = pd.DataFrame(test_df, columns=X_train.columns)
    
    
    # Train logistic regression model
    print("Training Logistic Regression...")
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    
    # Make predictions for logistic regression
    lin_train_predictions = model_lr.predict(X_train)
    lin_test_predictions = model_lr.predict(X_test)
    
    # Calculate accuracy scores for logistic regression
    lin_train_mape = mean_absolute_percentage_error(y_train, lin_train_predictions)
    lin_test_mape = mean_absolute_percentage_error(y_test, lin_test_predictions)
    lin_train_mse = mean_squared_error(y_train, lin_train_predictions)
    lin_test_mse = mean_squared_error(y_test, lin_test_predictions)
#     print("Logistic Regression")
    
#     print(f"Training MAPE: {lin_train_mape:.4f}")
#     print(f"Test MAPE: {lin_test_mape:.4f}")
    
#     print(f"Training MSE: {lin_train_mse:.4f}")
#     print(f"Test MSE: {lin_test_mse:.4f}")
    print()
    
    # Train KNN classification model
    print("Training KNN Regression...")
    model_knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    model_knn.fit(X_train, y_train)
    
    # Make predictions for KNN classification
    knn_train_predictions = model_knn.predict(X_train)
    knn_test_predictions = model_knn.predict(X_test)
    
    # Calculate accuracy scores for KNN classification
    knn_train_mape = mean_absolute_percentage_error(y_train, knn_train_predictions)
    knn_test_mape = mean_absolute_percentage_error(y_test, knn_test_predictions)
    knn_train_mse = mean_squared_error(y_train, knn_train_predictions)
    knn_test_mse = mean_squared_error(y_test, knn_test_predictions)
#     print("KNN Regression")
    
#     print(f"Training MAPE: {knn_train_mape:.4f}")
#     print(f"Test MAPE: {knn_test_mape:.4f}")
    
#     print(f"Training MSE: {knn_train_mse:.4f}")
#     print(f"Test MSE: {knn_test_mse:.4f}")
    print()
    
    # Train Decision Tree classification model
    print("Training Decision Tree Regression...")
    model_dt = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=leaf_nodes)
    model_dt.fit(X_train, y_train)
    
    # Make predictions for Decision Tree classification
    dt_train_predictions = model_dt.predict(X_train)
    dt_test_predictions = model_dt.predict(X_test)
    
    # Calculate accuracy scores for Decision Tree classification
    dt_train_mape = mean_absolute_percentage_error(y_train, dt_train_predictions)
    dt_test_mape = mean_absolute_percentage_error(y_test, dt_test_predictions)
    
    dt_train_mse = mean_squared_error(y_train, dt_train_predictions)
    dt_test_mse = mean_squared_error(y_test, dt_test_predictions)
    
#     print("Decision Tree Regression")
#     print(f"Training MAPE: {dt_train_mape:.4f}")
#     print(f"Test MAPE: {dt_test_mape:.4f}")
    
#     print(f"Training MSE: {dt_train_mse:.4f}")
#     print(f"Test MSE: {dt_test_mse:.4f}")
    print()
    
    
    model_performance= pd.DataFrame({'Name': ['Linear Regression_MAPE', 'Linear Regression_MSE', 'KNN_MAPE','KNN_MSE','Decision_Tree_MAPE','Decision_Tree_MSE'],
                                     'Train': [lin_train_mape, lin_train_mse,knn_train_mape,knn_train_mse,dt_train_mape,dt_train_mse], 
                                     'Test': [lin_test_mape, lin_test_mse,knn_test_mape,knn_test_mse,dt_test_mape,dt_test_mse]},columns=['Name','Train','Test'])
    return model_performance