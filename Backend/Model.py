# pylint: disable=import-error
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import tree
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, explained_variance_score, r2_score
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.preprocessing import MaxAbsScaler
import joblib
import sys

# get the command line arguments
args = sys.argv[1:]

# assign the arguments to the variables
actor_1_name = args[0]
actor_2_name = args[1]
actor_3_name = args[2]
director_name = args[3]
country = args[4]
content_rating = args[5]
language = args[6]
actor_1_facebook_likes = int(args[7])
actor_2_facebook_likes = int(args[8])
actor_3_facebook_likes = int(args[9])
director_facebook_likes = int(args[10])
cast_total_facebook_likes = int(args[11])
budget = float(args[12])
gross = float(args[13])
genres = args[14]
imdb_score = float(args[15])

# print out the values of the variables
print(f"actor_1_name: {actor_1_name}")
print(f"actor_2_name: {actor_2_name}")
print(f"actor_3_name: {actor_3_name}")
print(f"director_name: {director_name}")
print(f"country: {country}")
print(f"content_rating: {content_rating}")
print(f"language: {language}")
print(f"actor_1_facebook_likes: {actor_1_facebook_likes}")
print(f"actor_2_facebook_likes: {actor_2_facebook_likes}")
print(f"actor_3_facebook_likes: {actor_3_facebook_likes}")
print(f"director_facebook_likes: {director_facebook_likes}")
print(f"cast_total_facebook_likes: {cast_total_facebook_likes}")
print(f"budget: {budget}")
print(f"gross: {gross}")
print(f"genres: {genres}")
print(f"imdb_score: {imdb_score}")

f = open('english.txt', 'w')
f.write(actor_1_name)
f.close()


# In[3]:
global minval
global maxval
global min_max_scaler
global catagory_features
global number_features
global max_abs_scaler

max_abs_scaler = MaxAbsScaler()
min_max_scaler = preprocessing.MinMaxScaler()
RobustScaler = preprocessing.RobustScaler()
text_features = ['genre', 'plot_keywords', 'movie_title']
catagory_features = ['actor_1_name', 'actor_2_name', 'actor_3_name',
                     'director_name', 'country', 'content_rating', 'language']
number_features = ['actor_1_facebook_likes', 'actor_2_facebook_likes', 'actor_3_facebook_likes',
                   'director_facebook_likes', 'cast_total_facebook_likes', 'budget', 'gross']
all_selected_features = ['actor_1_name', 'actor_2_name', 'actor_3_name', 'director_name', 'country', 'content_rating', 'language', 'actor_1_facebook_likes',
                         'actor_2_facebook_likes', 'actor_3_facebook_likes', 'director_facebook_likes', 'cast_total_facebook_likes', 'budget', 'gross', 'genres', "imdb_score"]
eliminate_if_empty_list = ['actor_1_name', 'actor_2_name', 'director_name', 'country', 'actor_1_facebook_likes',
                           'actor_2_facebook_likes', 'director_facebook_likes', 'cast_total_facebook_likes', 'gross', "imdb_score"]

# preprocessing


def data_clean(path):
    read_data = pd.read_csv(path)
    select_data = read_data[all_selected_features]
    data = select_data.dropna(
        axis=0, how='any', subset=eliminate_if_empty_list)
    data = data.reset_index(drop=True)
    for x in catagory_features:
        data[x] = data[x].fillna('None').astype('category')
    for y in number_features:
        data[y] = data[y].fillna(0.0).astype(float)
    return data


def preprocessing_numerical_minmax(data):
    global min_max_scaler
    scaled_data = min_max_scaler.fit_transform(data)
    return scaled_data


def preprocessing_categorical(data):
    label_encoder = LabelEncoder()
    label_encoded_data = label_encoder.fit_transform(data)
    label_binarizer = preprocessing.LabelBinarizer()
    label_binarized_data = label_binarizer.fit_transform(label_encoded_data)
    return label_binarized_data


def preprocessing_text(data):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorized_text = tfidf_vectorizer.fit_transform(data)
    return tfidf_vectorized_text

# regression model training


def regression_without_cross_validation(model, train_data, train_target, test_data):
    model.fit(train_data, train_target)
    prediction = model.predict(test_data)
    return prediction


def regression_with_cross_validation(model, data, target, n_fold, model_name, pred_type):
    print(pred_type, " (Regression Model: ", model_name)
    cross_val_score_mean_abs_err = cross_val_score(
        model, data, target, scoring='neg_mean_absolute_error', cv=n_fold)
    print("\nCross Validation Score (Mean Absolute Error)        : \n", -
          cross_val_score_mean_abs_err)
    print("\nCross Validation Score (Mean Absolute Error) (Mean) : \n", -
          cross_val_score_mean_abs_err.mean())
    cross_val_score_mean_sqr_err = cross_val_score(
        model, data, target, scoring='neg_mean_squared_error', cv=n_fold)
    print("\nCross Validation Score (Mean Squared Error)         : \n", -
          cross_val_score_mean_sqr_err)
    print("\nCross Validation Score (Mean Squared Error)  (Mean) : \n", -
          cross_val_score_mean_sqr_err.mean())


def regression_scores(original_val, predicted_val, model_name):
    print("Regression Model Name: ", model_name)
    mean_abs_error = mean_absolute_error(original_val, predicted_val)
    mean_sqr_error = mean_squared_error(original_val, predicted_val)
    median_abs_error = median_absolute_error(original_val, predicted_val)
    explained_var_score = explained_variance_score(original_val, predicted_val)
    r2__score = r2_score(original_val, predicted_val)

    print("\n")
    print("\nRegression Scores(train_test_split):\n")
    print("Mean Absolute Error    :", mean_abs_error)
    print("Mean Squared Error     :", mean_sqr_error)
    print("Median Absolute Error  :", median_abs_error)
    print("Explained Var Score    :", explained_var_score)
    print("R^2 Score              :", r2__score)
    print("\n\n")

# simple task


def convert_gross(res):
    decimal_string = res[1:4]  # Extract first 5 characters of the string
    decimal_number = float(decimal_string)  # Convert string to float
    result = decimal_number * 2.7  # Multiply by 2.7
    return str(result)


def convert_score(res):
    decimal_string = res[1:4]  # Extract first 5 characters of the string
    decimal_number = float(decimal_string)  # Convert string to float
    result = decimal_number  # Multiply by 2.7
    return str(result)


def inverse_scaling(scaled_val):
    unscaled_val = min_max_scaler.inverse_transform(scaled_val.reshape(-1, 1))
    return unscaled_val


def roundval(value):
    return value.round()


def to_millions(value):
    return value / 10000000

# printing actual vs predicted in a range


def print_original_vs_predicted(original_val, predicted_val, i, j, n, model_name, print_type, prediction_type):
    # inverse transform and convert to millions
    original_val = to_millions(inverse_scaling(original_val))
    predicted_val = to_millions(inverse_scaling(predicted_val))
    print(predicted_val)
    print(predicted_val)
    ans = 0
    print("\n"+prediction_type +
          " Comparision of Actual VS Predicted"+print_type+"\n")
    if print_type == "seq":
        if j < len(predicted_val):
            for k in range(i, j + 1):
                print("Actual" + prediction_type+" : ",
                      original_val[k], ",   Predicted " + prediction_type, " : ", predicted_val[k])
                ans = predicted_val[k]
    if print_type == "random":
        for k in range(n):
            i = random.randint(0, len(predicted_val) - 1)
            print("Actual ", prediction_type, " : ",
                  original_val[i], ",   Predicted " + prediction_type+" : ", predicted_val[i])
            ans = predicted_val[i]
    return ans

# plotting actual vs predicted in a randomly using a bar chart


def meanbyfeature(data, feature_name, meanby_feature):
    mean_data = data.groupby(feature_name).mean()
    mean = mean_data[[meanby_feature]]
    mean_sort = mean.sort_values(
        by=[meanby_feature], axis=0, inplace=False, ascending=False)
    return mean_sort


def show_features(database):
    print("\n", "--------------------------------------------------------------------------------------------------------")
    database.info()
    print("\n", "--------------------------------------------------------------------------------------------------------")


# In[4]:

def preprocessing_catagory(data):
    data_c = 0
    for i in range(len(catagory_features)):
        new_data = data[catagory_features[i]]
        new_data_c = preprocessing_categorical(new_data)
        if i == 0:
            data_c = new_data_c
        else:
            data_c = np.append(data_c, new_data_c, 1)
    return data_c


def preprocessing_numerical(data):
    data_list_numerical = list(zip(data['director_facebook_likes'], data['actor_1_facebook_likes'],
                                   data['actor_2_facebook_likes'], data['actor_3_facebook_likes'],
                                   data['cast_total_facebook_likes'], data['budget']))

    data_numerical = np.array(data_list_numerical)
    data_numerical = preprocessing_numerical_minmax(data_numerical)
    return data_numerical


def preprocessed_agregated_data(database):
    numerical_data = preprocessing_numerical(database)
    categorical_data = preprocessing_catagory(database)
    all_data = np.append(numerical_data, categorical_data, 1)
    return all_data


def regr_without_cross_validation_train_test_perform_plot(model, data, target, model_name, pred_type):
    train_data = data[:-1]
    test_data = data[-1:]
    train_target = target[:-1]
    test_target = target[-1:]
    print("test data")
    print(test_data)
    print("test target")
    print(test_target)
    predicted_gross = regression_without_cross_validation(
        model, train_data, train_target, test_data)
    regression_scores(test_target, predicted_gross, model_name)
    # prediction_performance_plot(
    #     test_target, predicted_gross, model_name, 200, 250, 0, "seq", pred_type)
    # prediction_performance_plot(
    #     test_target, predicted_gross, model_name, 0, 0, 100, "random", pred_type)
    ans = print_original_vs_predicted(
        test_target, predicted_gross, 0, 0, 10, model_name, "random", pred_type)
    return ans


# In[5]:

path = "movie_metadata.csv"
data = data_clean(path)
list_new = [
    actor_1_name,
    actor_2_name,
    actor_3_name,
    director_name,
    country,
    content_rating,
    language,
    actor_1_facebook_likes,
    actor_2_facebook_likes,
    actor_3_facebook_likes,
    director_facebook_likes,
    cast_total_facebook_likes,
    budget,
    gross,
    genres,
    imdb_score
]
print(list_new)
data.drop(data.tail(1).index, inplace=True)
data.loc[len(data)] = list_new
#data = data[(data.actor_1_facebook_likes > 0.0) & (data.actor_2_facebook_likes > 0.0) & (data.actor_3_facebook_likes > 0.0) & (data.director_facebook_likes > 0.0) & (data.cast_total_facebook_likes > 0.0) & (data.gross > 0.0)]
target_gross = data['gross']
target_imdb_score = data['imdb_score']
database = data.drop('gross', axis=1)
database = data.drop('imdb_score', axis=1)
preprocessed_data = preprocessed_agregated_data(database)
target_gross = preprocessing_numerical_minmax(
    target_gross.values.reshape(-1, 1))
target_imdb_score = preprocessing_numerical_minmax(
    target_imdb_score.values.reshape(-1, 1))
print(target_gross)
print("feature calculation complete\n")
reg = joblib.load('regression_model.joblib')
predictions = reg.predict(preprocessed_data[-1:])
ans = print_original_vs_predicted(target_gross[-1:], predictions, 0, 0, 10,
                                  "Random Forest Regression", "random", "(Movie Gross Prediction)")
reg2 = joblib.load('regression_model_score.joblib')
predictions = reg2.predict(preprocessed_data[-1:])
ans2 = print_original_vs_predicted(target_imdb_score[-1:], predictions, 0, 0, 10,
                                   "Random Forest Regression", "random", "(IMDB Score Prediction)")

# In[ ]:

# Plotting


print("The Predicted Gross Revenue is:" + str(ans))
rfp = convert_gross(str(ans))
# dtp = convert_gross(str(ans2))
rfs = convert_score(str(ans2))
# dts = convert_score(str(ans4))

f = open("ans.txt", 'w')
f.write('$' + rfp + 'M' + '\n')
# f.write('$' + rfs + 'M' + '\n')
f.write(rfs + '\n')
# f.write(dts)
f.close()
# In[ ]:
