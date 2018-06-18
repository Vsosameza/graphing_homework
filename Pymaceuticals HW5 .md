

```python
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
%matplotlib inline 
clinical = "clinicaltrial_data.csv"
clinical_df = pd.read_csv(clinical)
mouse_drug = "mouse_drug_data.csv"
mousedrug_df = pd.read_csv(mouse_drug)
#clinical_df.head(5) 
#mousedrug_df.head(15)
#mousedrug_df.Drug.unique()
merged_tablesdf = pd.merge (clinical_df, mousedrug_df, on="Mouse ID", how="outer")
#merged_tablesdf.head(10)


```

Capomulin outperformed the infubinol, placebo and ketapril treatments in tumor volume reduction as well as in survivability. Capomulin also had a lower occurence of metastatic site occurences. 
Ketapril has a positive effect on tumor volume. Tumors of the mice treated with ketapril saw only slightly larger growth than the placebo group mice. 
The group of mice treated with infubinal experienced the lowest survival rate of the three other treatments. 


```python
print("\r\nTumor Response to Treatment")
```

    
    Tumor Response to Treatment
    


```python
#use pivot table P.315 compare all drug's aggregated av performance

merged_tablesdf_pivot_df = merged_tablesdf.pivot_table(values="Tumor Volume (mm3)",
                                                      index="Timepoint",
                                                      columns="Drug",
                                                      aggfunc="mean")
#only interested in Capomulin, Infubinol, Ketapril, Placebo 
#Sort out only these drugs out of the 10 
trial_drugs = ["Capomulin", "Infubinol", "Ketapril", "Placebo"]
trial_drugs_pivot_df = merged_tablesdf_pivot_df[trial_drugs]
trial_drugs_pivot_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Drug</th>
      <th>Capomulin</th>
      <th>Infubinol</th>
      <th>Ketapril</th>
      <th>Placebo</th>
    </tr>
    <tr>
      <th>Timepoint</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>45.000000</td>
      <td>45.000000</td>
      <td>45.000000</td>
      <td>45.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>44.266086</td>
      <td>47.062001</td>
      <td>47.389175</td>
      <td>47.125589</td>
    </tr>
    <tr>
      <th>10</th>
      <td>43.084291</td>
      <td>49.403909</td>
      <td>49.582269</td>
      <td>49.423329</td>
    </tr>
    <tr>
      <th>15</th>
      <td>42.064317</td>
      <td>51.296397</td>
      <td>52.399974</td>
      <td>51.359742</td>
    </tr>
    <tr>
      <th>20</th>
      <td>40.716325</td>
      <td>53.197691</td>
      <td>54.920935</td>
      <td>54.364417</td>
    </tr>
    <tr>
      <th>25</th>
      <td>39.939528</td>
      <td>55.715252</td>
      <td>57.678982</td>
      <td>57.482574</td>
    </tr>
    <tr>
      <th>30</th>
      <td>38.769339</td>
      <td>58.299397</td>
      <td>60.994507</td>
      <td>59.809063</td>
    </tr>
    <tr>
      <th>35</th>
      <td>37.816839</td>
      <td>60.742461</td>
      <td>63.371686</td>
      <td>62.420615</td>
    </tr>
    <tr>
      <th>40</th>
      <td>36.958001</td>
      <td>63.162824</td>
      <td>66.068580</td>
      <td>65.052675</td>
    </tr>
    <tr>
      <th>45</th>
      <td>36.236114</td>
      <td>65.755562</td>
      <td>70.662958</td>
      <td>68.084082</td>
    </tr>
  </tbody>
</table>
</div>




```python
#data to grab standard error of the mean:
std_err_volume_pivot_df = merged_tablesdf.pivot_table(values="Tumor Volume (mm3)",
                                                      index="Timepoint",
                                                      columns="Drug",
                                                      aggfunc="sem")
trial_drugs_std_err_df = std_err_volume_pivot_df[trial_drugs]
trial_drugs_std_err_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Drug</th>
      <th>Capomulin</th>
      <th>Infubinol</th>
      <th>Ketapril</th>
      <th>Placebo</th>
    </tr>
    <tr>
      <th>Timepoint</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.448593</td>
      <td>0.235102</td>
      <td>0.264819</td>
      <td>0.218091</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.702684</td>
      <td>0.282346</td>
      <td>0.357421</td>
      <td>0.402064</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.838617</td>
      <td>0.357705</td>
      <td>0.580268</td>
      <td>0.614461</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.909731</td>
      <td>0.476210</td>
      <td>0.726484</td>
      <td>0.839609</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.881642</td>
      <td>0.550315</td>
      <td>0.755413</td>
      <td>1.034872</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0.934460</td>
      <td>0.631061</td>
      <td>0.934121</td>
      <td>1.218231</td>
    </tr>
    <tr>
      <th>35</th>
      <td>1.052241</td>
      <td>0.984155</td>
      <td>1.127867</td>
      <td>1.287481</td>
    </tr>
    <tr>
      <th>40</th>
      <td>1.223608</td>
      <td>1.055220</td>
      <td>1.158449</td>
      <td>1.370634</td>
    </tr>
    <tr>
      <th>45</th>
      <td>1.223977</td>
      <td>1.144427</td>
      <td>1.453186</td>
      <td>1.351726</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig=plt.figure(figsize=(10,8))
ax=fig.add_subplot(111)
drug_performance = []
for column, values in trial_drugs_pivot_df.iteritems():
    drug_performance.append(list(values))
drug_vol_sem = []
for column, values in trial_drugs_std_err_df.iteritems():
    drug_vol_sem.append(list(values))
x_axis = list(trial_drugs_pivot_df.index)
markers = ['o', '^', 's','D']
colors = ['black', 'green', 'orange', 'purple']
for marker, drugs, data, sem, color in zip(markers, trial_drugs, drug_performance, drug_vol_sem, colors):
    ax.plot(x_axis, 
            data,
            marker=marker,
            label=drugs,
            c=color,
            linestyle='--')

    ax.errorbar(x_axis, 
                data, 
                yerr=sem, 
                c=color, 
                capsize=3, 
                linestyle='--')
plt.xlim(0, 45)
plt.ylim(20, 80)
plt.title('Tumor Response to Treatment')
plt.xlabel('Time (Days)')
plt.ylabel('Tumor Volumes (mm3)')
plt.grid(linestyle='dashed')
plt.legend()
plt.show()

print("\r\nMetastatic Response to Treatment")
```


![png](output_5_0.png)


    
    Metastatic Response to Treatment
    


```python
trial_metastatic_pivot_df = merged_tablesdf.pivot_table(values='Metastatic Sites', 
                                                     index='Timepoint', 
                                                     columns='Drug', 
                                                     aggfunc='mean')
selected_drugs_meta_pivot_df = trial_metastatic_pivot_df[trial_drugs]
selected_drugs_meta_pivot_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Drug</th>
      <th>Capomulin</th>
      <th>Infubinol</th>
      <th>Ketapril</th>
      <th>Placebo</th>
    </tr>
    <tr>
      <th>Timepoint</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.160000</td>
      <td>0.280000</td>
      <td>0.304348</td>
      <td>0.375000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.320000</td>
      <td>0.666667</td>
      <td>0.590909</td>
      <td>0.833333</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.375000</td>
      <td>0.904762</td>
      <td>0.842105</td>
      <td>1.250000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.652174</td>
      <td>1.050000</td>
      <td>1.210526</td>
      <td>1.526316</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.818182</td>
      <td>1.277778</td>
      <td>1.631579</td>
      <td>1.941176</td>
    </tr>
    <tr>
      <th>30</th>
      <td>1.090909</td>
      <td>1.588235</td>
      <td>2.055556</td>
      <td>2.266667</td>
    </tr>
    <tr>
      <th>35</th>
      <td>1.181818</td>
      <td>1.666667</td>
      <td>2.294118</td>
      <td>2.642857</td>
    </tr>
    <tr>
      <th>40</th>
      <td>1.380952</td>
      <td>2.100000</td>
      <td>2.733333</td>
      <td>3.166667</td>
    </tr>
    <tr>
      <th>45</th>
      <td>1.476190</td>
      <td>2.111111</td>
      <td>3.363636</td>
      <td>3.272727</td>
    </tr>
  </tbody>
</table>
</div>




```python
trial_sem_metastic_pivot_df = merged_tablesdf.pivot_table(values='Metastatic Sites', 
                                                     index='Timepoint', 
                                                     columns='Drug', 
                                                     aggfunc='sem')
trial_drugs_meta_sem_df = trial_sem_metastic_pivot_df[trial_drugs]
trial_drugs_meta_sem_df

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Drug</th>
      <th>Capomulin</th>
      <th>Infubinol</th>
      <th>Ketapril</th>
      <th>Placebo</th>
    </tr>
    <tr>
      <th>Timepoint</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.074833</td>
      <td>0.091652</td>
      <td>0.098100</td>
      <td>0.100947</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.125433</td>
      <td>0.159364</td>
      <td>0.142018</td>
      <td>0.115261</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.132048</td>
      <td>0.194015</td>
      <td>0.191381</td>
      <td>0.190221</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.161621</td>
      <td>0.234801</td>
      <td>0.236680</td>
      <td>0.234064</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.181818</td>
      <td>0.265753</td>
      <td>0.288275</td>
      <td>0.263888</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0.172944</td>
      <td>0.227823</td>
      <td>0.347467</td>
      <td>0.300264</td>
    </tr>
    <tr>
      <th>35</th>
      <td>0.169496</td>
      <td>0.224733</td>
      <td>0.361418</td>
      <td>0.341412</td>
    </tr>
    <tr>
      <th>40</th>
      <td>0.175610</td>
      <td>0.314466</td>
      <td>0.315725</td>
      <td>0.297294</td>
    </tr>
    <tr>
      <th>45</th>
      <td>0.202591</td>
      <td>0.309320</td>
      <td>0.278722</td>
      <td>0.304240</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig=plt.figure(figsize=(10,8))
ax=fig.add_subplot(111)

drug_meta_performance = []
for column, values in selected_drugs_meta_pivot_df.iteritems():
    drug_meta_performance.append(list(values))

    drug_meta_sem = []
for column, values in trial_drugs_meta_sem_df.iteritems():
    drug_meta_sem.append(list(values))

x_axis = list(selected_drugs_meta_pivot_df.index)
markers = ['o', '^', 's','D']
colors = ['black', 'green', 'orange', 'purple']

for marker, drugs, data, sem, color in zip(markers, trial_drugs, drug_meta_performance, drug_meta_sem, colors):
    ax.plot(x_axis, 
            data,
            marker=marker,
            label=drugs,
            c=color,
            linestyle='--')

    ax.errorbar(x_axis, 
                data, 
                yerr=sem, 
                c=color, 
                capsize=3, 
                linestyle='--')
plt.xlim(0, 45)
plt.ylim(0, 4)
plt.title('Metastatic Spread During Treatment')
plt.xlabel('Treatment Duration (Days)')
plt.ylabel('Met. Sites')
plt.grid(linestyle='dashed')
plt.legend()
plt.show()
print("\r\nSurvival Rates")
```


![png](output_8_0.png)


    
    Survival Rates
    


```python
trial_survival_pivot_df = merged_tablesdf.pivot_table(values='Mouse ID', 
                                                     index='Timepoint', 
                                                     columns='Drug', 
                                                     aggfunc='count')
trial_drugs_survival_df = trial_survival_pivot_df[trial_drugs]
#trial_drugs_survival_df

def normalize_mice_survival_rate(mouse_count):
    initial_mice_count = trial_drugs_survival_df.iloc[0].max()
    return (mouse_count / initial_mice_count) * 100

trial_drugs_survival_df = trial_drugs_survival_df.apply(normalize_mice_survival_rate)
trial_drugs_survival_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Drug</th>
      <th>Capomulin</th>
      <th>Infubinol</th>
      <th>Ketapril</th>
      <th>Placebo</th>
    </tr>
    <tr>
      <th>Timepoint</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100.0</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>100.0</td>
      <td>100.0</td>
      <td>92.0</td>
      <td>96.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>100.0</td>
      <td>84.0</td>
      <td>88.0</td>
      <td>96.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>96.0</td>
      <td>84.0</td>
      <td>76.0</td>
      <td>80.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>92.0</td>
      <td>80.0</td>
      <td>76.0</td>
      <td>76.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>88.0</td>
      <td>72.0</td>
      <td>76.0</td>
      <td>68.0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>88.0</td>
      <td>68.0</td>
      <td>72.0</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>88.0</td>
      <td>48.0</td>
      <td>68.0</td>
      <td>56.0</td>
    </tr>
    <tr>
      <th>40</th>
      <td>84.0</td>
      <td>40.0</td>
      <td>60.0</td>
      <td>48.0</td>
    </tr>
    <tr>
      <th>45</th>
      <td>84.0</td>
      <td>36.0</td>
      <td>44.0</td>
      <td>44.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig=plt.figure(figsize=(10,8))
ax=fig.add_subplot(111)

mouse_survival = []
for column, values in trial_drugs_survival_df.iteritems():
    mouse_survival.append(list(values))


x_axis = list(trial_drugs_survival_df.index)
markers = ['o', '^', 's','D']
colors = ['black', 'green', 'orange', 'purple']

for marker, drugs, data, color in zip(markers, trial_drugs, mouse_survival, colors):
    ax.plot(x_axis, 
            data,
            marker=marker,
            label=drugs,
            c=color,
            linestyle='--')

    ax.errorbar(x_axis, 
                data, 
                yerr=sem, 
                c=color, 
                capsize=3, 
                linestyle='--')
plt.xlim(0, 45)
plt.ylim(0, 100)
plt.title('Survival of Mice During Treatment')
plt.xlabel(' Duration (Days)')
plt.ylabel('Survival rate %')
plt.grid(linestyle='dashed')
plt.legend()
plt.show()
print("\r\nSummary Bar Graph")
```


![png](output_10_0.png)


    
    Summary Bar Graph
    


```python
trial_drugs_survival_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Drug</th>
      <th>Capomulin</th>
      <th>Infubinol</th>
      <th>Ketapril</th>
      <th>Placebo</th>
    </tr>
    <tr>
      <th>Timepoint</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100.0</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>100.0</td>
      <td>100.0</td>
      <td>92.0</td>
      <td>96.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>100.0</td>
      <td>84.0</td>
      <td>88.0</td>
      <td>96.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>96.0</td>
      <td>84.0</td>
      <td>76.0</td>
      <td>80.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>92.0</td>
      <td>80.0</td>
      <td>76.0</td>
      <td>76.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>88.0</td>
      <td>72.0</td>
      <td>76.0</td>
      <td>68.0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>88.0</td>
      <td>68.0</td>
      <td>72.0</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>88.0</td>
      <td>48.0</td>
      <td>68.0</td>
      <td>56.0</td>
    </tr>
    <tr>
      <th>40</th>
      <td>84.0</td>
      <td>40.0</td>
      <td>60.0</td>
      <td>48.0</td>
    </tr>
    <tr>
      <th>45</th>
      <td>84.0</td>
      <td>36.0</td>
      <td>44.0</td>
      <td>44.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#groupby drug for change in tumor size calcs>>>for bar chart
avg_tumor_size = pd.DataFrame(merged_tablesdf.groupby(['Drug', 'Timepoint']).mean()['Tumor Volume (mm3)'])
avg_tumor_size = avg_tumor_size.unstack(level = 0)
avg_tumor_size.columns = avg_tumor_size.columns.get_level_values(1)


change_in_tumor_size = (avg_tumor_size.loc[45, :] - avg_tumor_size.loc[0, :])/avg_tumor_size.loc[0, :] * 100
change_in_tumor_size
```




    Drug
    Capomulin   -19.475303
    Ceftamin     42.516492
    Infubinol    46.123472
    Ketapril     57.028795
    Naftisol     53.923347
    Placebo      51.297960
    Propriva     47.241175
    Ramicane    -22.320900
    Stelasyn     52.085134
    Zoniferol    46.579751
    dtype: float64




```python
plt.title('Tumor Volume Change over 45 Day Treatment')
plt.ylabel('Survival Rate (%)')
plt.axhline(y=0, color = 'black')
xlabels = change_in_tumor_size.index
plt.xticks(np.arange(len(xlabels)), xlabels, rotation = 90)
count = 0
plt.grid(linestyle='dashed')

height = change_in_tumor_size

bars = change_in_tumor_size.index

y_pos = np.arange(len(bars))

plt.bar(y_pos, height, 
        color = ['red' if change_in_tumor_size[r] > 0 else 'green' 
                 for r in np.arange(len(xlabels))])




```




    <BarContainer object of 10 artists>




![png](output_13_1.png)



```python



```
