def test_load():
  return 'loaded'

def compute_probs(neg,pos):
  p0 = neg/(neg+pos)
  p1 = pos/(neg+pos)
  return [p0,p1]

def cond_prob(table,evidence,evidence_value,target,target_value):
  t_subset = up_table_subset(table, target, 'equals', target_value)
  e_list = up_get_column(t_subset, evidence)
  p_b_a = sum([1 if v==evidence_value else 0 for v in e_list])/len(e_list)
  return p_b_a + .01

def cond_probs_product(table, evidence_row, target, target_value):
  table_columns=up_list_column_names(table)
  evidence_columns=table_columns[:-1]
  evidence_zipped=up_zip_lists(evidence_columns,evidence_row)
  cond_prob_list = [cond_prob(table,e_col,e_val,target,target_value) for e_col,e_val in evidence_zipped]
  partial_numerator = up_product(cond_prob_list)
  return partial_numerator

def prior_prob(table,target,target_value):
  t_list = up_get_column(table, target)
  p_a = sum([1 if v==target_value else 0 for v in t_list])/len(t_list)
  return p_a

def naive_bayes(table, evidence_row, target):
  final_1=[]
  final_2=[]
  target_val=0
  final_1=cond_probs_product(table, evidence_row, target, target_val)*prior_prob(table,target,target_val)
  target_val=1
  final_2=cond_probs_product(table, evidence_row, target, target_val)*prior_prob(table,target,target_val)
  p0,p1=compute_probs(final_1,final_2)
  return [p0, p1]

def metrics(zipped_list):
  #asserts here
  assert isinstance(zipped_list, list), "parameter should be a list."
  for metric in zipped_list:
    assert isinstance(metric, list), "The parameter should be a list of lists."
  assert all(len(metric) == 2 for metric in zipped_list), "Each element in metrics should be a pair."
  #assert len(metric) == len(zipped_list[0]), "The should be a zipped list."
  for metric in zipped_list:
    for value in metric:
       assert isinstance(value, int) and value >= 0, "Each value in the pair should be an integer that is non-negative."
  #body of function below
  tn = sum([1 if pair==[0,0] else 0 for pair in zipped_list])
  tp = sum([1 if pair==[1,1] else 0 for pair in zipped_list])
  fp = sum([1 if pair==[1,0] else 0 for pair in zipped_list])
  fn = sum([1 if pair==[0,1] else 0 for pair in zipped_list])
  
  if tp+fp == 0:
    precision = 0
  else:
    precision = tp /(tp+fp)
  
  if tp+fn == 0:
    recall = 0
  else:
    recall = tp /(tp+fn)
  
  if precision * recall == 0:
    f1 = 0
  else:
    f1 = 2*(precision * recall) / (precision + recall)
    
  accuracy = sum([p == a for p, a in zipped_list]) / len(zipped_list)  
  dictionary={'Precision': precision, 'Recall': recall, 'F1': f1,'Accuracy': accuracy}
  return dictionary

from sklearn.ensemble import RandomForestClassifier  #make sure this makes it into your library
def run_random_forest(train, test, target, n):
  #target is target column name
  #n is number of trees to use
  X = up_drop_column(train, target)
  y = up_get_column(train,target)  

  k_feature_table = up_drop_column(test, target)
  k_actuals = up_get_column(test, target)  

  clf = RandomForestClassifier(n_estimators=n, max_depth=2, random_state=0)

  clf.fit(X, y)  #builds the trees as specified above
  probs = clf.predict_proba(k_feature_table)
  pos_probs = [p for n,p in probs]  #probs is list of [neg,pos] like we are used to seeing.

  all_mets = []
  for t in thresholds:
    all_predictions = [1 if pos>t else 0 for pos in pos_probs]
    pred_act_list = up_zip_lists(all_predictions, k_actuals)
    mets = metrics(pred_act_list)
    mets['Threshold'] = t
    all_mets = all_mets + [mets]

  metrics_table = up_metrics_table(all_mets)
  return metrics_table
