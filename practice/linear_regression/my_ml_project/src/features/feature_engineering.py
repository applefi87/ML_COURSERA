import os
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

def feature_engineering_data(input_path,output_path,column_name="size"):
  """Main function to clean data."""
  data = pd.read_csv(input_path)
  poly = PolynomialFeatures(degree=1,include_bias=False)
  poly_data=poly.fit_transform(data[["size"]])
  poly_df=pd.DataFrame(poly_data,columns=poly.get_feature_names_out(["size"]))
  poly_df = poly_df.drop("size", axis=1)
  data_poly = pd.concat([data, poly_df], axis=1)
  data.to_csv(output_path,index=False)
  return data

#The if __name__ == "__main__": block allows you to run the script as a standalone file, 
# but you can also import and use the functions elsewhere without executing the whole script
if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.dirname(os.getcwd()))
    input_path = os.path.join(project_dir, 'data', 'interim', 'cleaned_data.csv')
    output_path = os.path.join(project_dir, 'data', 'processed', 'features_engineered.csv')
    feature_engineering_data(input_path, output_path)