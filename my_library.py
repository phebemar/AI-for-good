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
