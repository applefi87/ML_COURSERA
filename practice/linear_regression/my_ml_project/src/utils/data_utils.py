def convert_to_float(df, exclude_columns=[]):
    """Convert all numeric columns to float64, excluding specified columns."""
    cols = df.select_dtypes(include=['int32', 'int64']).columns.difference(exclude_columns)
    df[cols] = df[cols].astype('float64')
    return df