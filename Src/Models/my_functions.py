

import pandas as pd
import numpy as np
from scipy import stats

def handle_missing(dataframe):
    import pandas as pd
    missing_data_cols = dataframe.columns[dataframe.isnull().sum() > 0]
    null = dataframe.isnull().sum()
    null_df = pd.DataFrame(null, columns=["No_of_null_values"])
    null_df = null_df.loc[missing_data_cols]
    
    if len(null_df) > 0:
        for col in null_df.index:
            if dataframe[col].dtype == 'float64' or dataframe[col].dtype == 'int64':
                dataframe[col].fillna(dataframe[col].mode(), inplace=True)
            else:
                dataframe[col].fillna(dataframe[col].mode()[0], inplace=True)
        
        return dataframe, null_df
    else:
        return "There are no null values!"
    

def handle_outliers(df):
    from scipy import stats
    import numpy as np
    numerical_cols = df.select_dtypes("number").drop(['PerformanceRating'],axis = 1)
    z = np.abs(stats.zscore(numerical_cols))
    threshold = 3
    outliers = np.where(z > threshold)
    print(f"Using 'Z-Score Index we found there are {len(outliers[0])} Outliers")
    
    for col in numerical_cols.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    z = np.abs(stats.zscore(df[numerical_cols.columns]))
    threshold = 3
    outliers = np.where(z > threshold)
    print(f"After handling outlier using IQR method we found there are {len(outliers[0])} Outliers")


def encode_cat_feats(df):
    cat_cols = df.select_dtypes("object")
    cat_cols_list = cat_cols.columns.to_list()
    encoded_df = pd.get_dummies(df, columns=cat_cols_list, drop_first=True)
    performance_rating_column = encoded_df.pop('PerformanceRating')
    encoded_df['PerformanceRating'] = performance_rating_column
    encoded_df['PerformanceRating'] = encoded_df['PerformanceRating'].map({2: 0, 3: 1, 4: 2})
    return encoded_df

def scale_data(df, scaler=None):

    from sklearn.preprocessing import MinMaxScaler

    numerical_cols = df.select_dtypes(include=['int64', 'float64'])

    if scaler is None:
        scaler = MinMaxScaler()
        scaled_numerical_cols = scaler.fit_transform(numerical_cols)
    else:
        scaled_numerical_cols = scaler.transform(numerical_cols)

    scaled_df = pd.DataFrame(scaled_numerical_cols, columns=numerical_cols.columns)

    encoded_categorical_cols = df.select_dtypes(include=['uint8']).reset_index(drop=True)

    final_X = pd.concat([scaled_df, encoded_categorical_cols], axis=1)

    return final_X