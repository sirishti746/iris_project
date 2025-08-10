from loguru import logger
from constants import DATA_FILE, MODEL_DIR, MODEL_FILE, TEST_SIZE, RANDOM_STATE,TARGET
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score , classification_report
import joblib

def train_and_save_model():
    try:
        #STEP 1 - DATA INJESTION
        logger.info(f"Performing data ingestion on {DATA_FILE}...")
        df = pd.read_csv(DATA_FILE)
        logger.info(f"Data shape: {df.shape}")
        logger.info(f"data columns: {df.columns.to_list()}")
        
        #STEP2 - REMOVE DUPLICATES
        dup = df.duplicated().sum()
        logger.info(f"Number of duplicate rows: {dup}")
        df = df.drop_duplicates(keep="first").reset_index(drop=True)
        logger.info(f"Duplicates dropped, data shape: {df.shape}")
        
        # check for missing values
        m = df.isna().sum()
        logger.info(f"missing values: \n{m.to_dict()}")
        #Separate x and y
        logger.info("separating x and  y")
        X = df.drop(columns=[TARGET])
        Y = df[TARGET]
        
        #Apply train test split
        logger.info("applying train test split")    
        X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=TEST_SIZE,random_state=RANDOM_STATE)
        logger.info(f"xtrain shape: {X_train.shape},ytrain shape: {y_train.shape}")
        logger.info(f"xtest shape : {X_test.shape}, ytest shape : {y_test.shape}" )
        
        #Initialise pipeline model
        logger.info("initialising pipeline model")
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(random_state=RANDOM_STATE)
        )
        
        #Cross vlidate the model
        scores = cross_val_score(model,X_train,y_train,cv=5,scoring="f1_macro")
        logger.info(f"cross validation scores: {scores}")
        scores_mean = scores.mean().round(4)
        scores_std = scores.std().round(4)
        logger.info(f"Mean cross val score : {scores_mean} +/- {scores_std}")
        
        #train the model
        logger.info("training the model")
        model.fit(X_train, y_train)
        logger.info("model training done!!")
        
        #model evaluation
        logger.info("model evaluation") 
        ypred_train = model.predict(X_train)
        ypred_test = model.predict(X_test)
        f1_train = f1_score(y_train,ypred_train,average="macro")
        f1_test = f1_score(y_test,ypred_test,average="macro")
        logger.info(f"train f1 score: {f1_train:.4f}")
        logger.info(f"F1 macro test: {f1_test:.4f} ")
        logger.info(f"Classification report on test data: {classification_report(y_test,ypred_test)} ")
        
        #save the model 
        logger.info(f"saving the model object to : {MODEL_FILE}")
        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(model,MODEL_FILE)
        logger.info(f"model saved!!")
        
        logger.success("Training pipeline successful!")
        
    except Exception as e:
        logger.error(f"error occured: {e}")
        
        


if __name__ == "__main__":
    train_and_save_model()
