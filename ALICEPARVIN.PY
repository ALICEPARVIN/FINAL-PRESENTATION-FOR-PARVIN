
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


dataset = [
    ["Bread", "Milk", "Eggs"],
    ["Bread", "Butter"],
    ["Milk", "Diapers", "Beer"],
    ["Bread", "Milk", "Butter"],
    ["Milk", "Diapers", "Bread"],
    ["Beer", "Diapers"],
    ["Bread", "Milk", "Eggs", "Butter"],
    ["Eggs", "Milk"],
    ["Bread", "Diapers", "Beer"],
    ["Milk", "Butter"]
]


te = TransactionEncoder()
te_data = te.fit(dataset).transform(dataset)


df = pd.DataFrame(te_data, columns=te.columns_)
print("\nONE-HOT ENCODED DATAFRAME:")
print(df)



frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)
print("\nFREQUENT ITEMSETS (support >= 0.2):")
print(frequent_itemsets)


rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)


rules = rules[["antecedents", "consequents", "support", "confidence", "lift"]]
print("\nASSOCIATION RULES:")
print(rules)

