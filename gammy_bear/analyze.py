class analyzer():
  def __init__(self, df):
    self.df = df
    self

  def numer_of_instances(self, df=None):
    if df is None:
      self.numer_of_instances = len(self.df) 
      return self.numer_of_instances
    return len(df)
  
  def numer_of_features(self, df=None):
      if df is None:
        self.numer_of_features = len(self.df.columns)
        return self.numer_of_features
      return len(df.columns)

  def total_of_classes(self, df):
    unique = {}
    if df is None:
      df = self.df
    for col in df.columns:
      unique[col] = len(df[col].unique())
    self.unique = unique
    return unique
  def class_distribution(df):
      results = {}
      for col in df.columns:
          results[col] = dict(df[col].value_counts())
      return results