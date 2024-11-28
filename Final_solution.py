import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

X_train = pd.read_csv('/Users/wudongyan/Desktop/mini-project/X_train.csv')
y_train = pd.read_csv('/Users/wudongyan/Desktop/mini-project/y_train.csv')
X_test = pd.read_csv('/Users/wudongyan/Desktop/mini-project/X_test.csv')

# 合併 X_train 和 y_train
train_data = pd.merge(X_train, y_train, on='Id', how='inner')

# 處理日期，計算房齡
train_data['建築完成年'] = pd.to_datetime(train_data['建築完成年月'], errors='coerce').dt.year
X_test['建築完成年'] = pd.to_datetime(X_test['建築完成年月'], errors='coerce').dt.year

train_data['房齡'] = 2024 - train_data['建築完成年']
X_test['房齡'] = 2024 - X_test['建築完成年']

train_data['房齡'].fillna(0, inplace=True)
X_test['房齡'].fillna(0, inplace=True)

# 特徵交互
train_data['房齡_總樓層'] = train_data['房齡'] * train_data['總樓層數']
train_data['房齡_建物移轉總面積'] = train_data['房齡'] * train_data['建物移轉總面積平方公尺']

X_test['房齡_總樓層'] = X_test['房齡'] * X_test['總樓層數']
X_test['房齡_建物移轉總面積'] = X_test['房齡'] * X_test['建物移轉總面積平方公尺']

# 類別變數編碼
categorical_features = ['鄉鎮市區', '交易標的', '路名', '都市土地使用分區', '移轉層次項目', '建物型態', '主要用途', '主要建材', '有無管理組織', '建物現況格局-隔間']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

encoded_train = encoder.fit_transform(train_data[categorical_features])
encoded_test = encoder.transform(X_test[categorical_features])

X_train_encoded = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out())
X_test_encoded = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out())

train_data.drop(columns=categorical_features + ['建築完成年月'], inplace=True)
X_test.drop(columns=categorical_features + ['建築完成年月'], inplace=True)

X_train_final = pd.concat([train_data.reset_index(drop=True), X_train_encoded], axis=1)
X_test_final = pd.concat([X_test.reset_index(drop=True), X_test_encoded], axis=1)

y_train_final = train_data['單價元平方公尺']
X_train_final.drop(columns=['Id', '單價元平方公尺'], inplace=True)
X_test_final.drop(columns=['Id'], inplace=True)

# 使用 K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 定義超參數搜尋範圍
xgb_param_grid = {
    'n_estimators': [500, 1000, 1500],
    'max_depth': [6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0, 0.1, 0.5]
}

lgb_param_grid = {
    'n_estimators': [500, 1000, 1500],
    'max_depth': [6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0, 0.1, 0.5]
}

# 執行超參數調整
xgb_model = XGBRegressor(random_state=42)
lgb_model = LGBMRegressor(random_state=42)

xgb_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=xgb_param_grid,
    n_iter=20,
    scoring='neg_root_mean_squared_error',
    cv=5,
    n_jobs=-1,
    random_state=42
)

lgb_search = RandomizedSearchCV(
    estimator=lgb_model,
    param_distributions=lgb_param_grid,
    n_iter=20,
    scoring='neg_root_mean_squared_error',
    cv=5,
    n_jobs=-1,
    random_state=42
)

print("調整 XGBoost 模型超參數中...")
xgb_search.fit(X_train_final, y_train_final)
print("Best XGBoost Parameters:", xgb_search.best_params_)

print("調整 LightGBM 模型超參數中...")
lgb_search.fit(X_train_final, y_train_final)
print("Best LightGBM Parameters:", lgb_search.best_params_)


best_xgb = xgb_search.best_estimator_
best_lgb = lgb_search.best_estimator_

y_pred_test_xgb = best_xgb.predict(X_test_final)
y_pred_test_lgb = best_lgb.predict(X_test_final)

# 平均預測結果
y_pred_test = (y_pred_test_xgb + y_pred_test_lgb) / 2
submission = pd.DataFrame({'Id': X_test['Id'], '單價元平方公尺': y_pred_test})
submission.to_csv('submission_tuned.csv', index=False)
print("提交檔案已生成：submission_tuned.csv")