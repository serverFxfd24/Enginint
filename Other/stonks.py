# Анализ изменений цены акций (не используется)
import pandas as pd
from datetime import datetime, timedelta


# Кластер акций
#### CHMF
CHMF = pd.read_csv('/content/sample_data/CHMF Акции.csv')
CHMF = CHMF[len(CHMF)::-1]
CHMF.reset_index(drop = True,inplace=True)
CHMF_date = CHMF['Date']
CHMF_change = CHMF['Change %']
CHMF_change = CHMF_change.str.replace('%', '')
CHMF_change = CHMF_change.astype(float)
CHMF_change = list(CHMF_change)
CHMF_date = list(CHMF_date)
CHMF_week = []
mounth = [0]

for i in range(0,len(CHMF_date)):
  if CHMF_date[i] == '01/03/2019':
    pass
  elif (list(CHMF_date[i-1])[0] == list(CHMF_date[i])[0]):
    if (list(CHMF_date[i-1])[1] != list(CHMF_date[i])[1]):
      mounth.append(i)
  else:
    mounth.append(i)

for i in range(1,len(mounth)):
  CHMF_week.append(sum(CHMF_change[mounth[i-1]:mounth[i]])/len(CHMF_change[mounth[i-1]:mounth[i]]))

#### NLMK
NLMK = pd.read_csv('/content/sample_data/NLMK Акции.csv')
NLMK = NLMK[len(NLMK)::-1]
NLMK.reset_index(drop = True,inplace=True)
NLMK_date = NLMK['Date']
NLMK_change = NLMK['Change %']
NLMK_change = NLMK_change.str.replace('%', '')
NLMK_change = NLMK_change.astype(float)
NLMK_change = list(NLMK_change)
NLMK_date = list(NLMK_date)
NLMK_week = []
mounth = [0]

for i in range(0,len(NLMK_date)):
  if NLMK_date[i] == '01/03/2019':
    pass
  elif (list(NLMK_date[i-1])[0] == list(NLMK_date[i])[0]):
    if (list(NLMK_date[i-1])[1] != list(NLMK_date[i])[1]):
      mounth.append(i)
  else:
    mounth.append(i)

for i in range(1,len(mounth)):
  NLMK_week.append(sum(NLMK_change[mounth[i-1]:mounth[i]])/len(NLMK_change[mounth[i-1]:mounth[i]]))


#### MAGN
MAGN = pd.read_csv('/content/sample_data/MAGN Акции.csv')
MAGN = MAGN[len(MAGN)::-1]
MAGN.reset_index(drop = True,inplace=True)
MAGN_date = MAGN['Дата']
MAGN_date = MAGN_date.str.replace('.', '/')
M_=[]
for k in MAGN_date:
  i = list(k)
  i[0], i[3] = i[3], i[0]
  i[1], i[4] = i[4], i[1]
  M_.append(''.join(i))
MAGN_date = M_
MAGN_change = MAGN['Изм. %']
MAGN_change = MAGN_change.str.replace('%', '')
MAGN_change = MAGN_change.str.replace(',', '.')
MAGN_change = MAGN_change.astype(float)
MAGN_change = list(MAGN_change)
MAGN_date = list(MAGN_date)
MAGN_week = []
mounth = [0]

for i in range(0,len(MAGN_date)):
  if MAGN_date[i] == '01/03/2019':
    pass
  elif (list(MAGN_date[i-1])[0] == list(MAGN_date[i])[0]):
    if (list(MAGN_date[i-1])[1] != list(MAGN_date[i])[1]):
      mounth.append(i)
  else:
    mounth.append(i)

for i in range(1,len(mounth)):
  MAGN_week.append(sum(MAGN_change[mounth[i-1]:mounth[i]])/len(MAGN_change[mounth[i-1]:mounth[i]]))


### нормальные значения времени через библиотечку
start_date = datetime(2019, 1, 7)
end_date = datetime(2023, 3, 7)
date_list = [start_date + timedelta(weeks=i) for i in range(int((end_date - start_date).days / 7))]

week_counts = {}
week_counts_ = []

for date in date_list:
    year_month = date.strftime("%Y-%m")
    week_counts.setdefault(year_month, 0)
    week_counts[year_month] += 1

for year_month, count in week_counts.items():
    year, month = year_month.split("-")
    month_name = datetime.strptime(month, "%m").strftime("%B")
    # print(f"In {month_name} {year}, there are {count} weeks.")
    week_counts_.append(count)

# print(week_counts_)
CHMF_week_ = []
NLMK_week_ = []
MAGN_week_ = []
for i,k in zip(CHMF_week,week_counts_):
  if k == 4:
    CHMF_week_.append(i)
    CHMF_week_.append(i)
    CHMF_week_.append(i)
    CHMF_week_.append(i)
  elif k == 5:
    CHMF_week_.append(i)
    CHMF_week_.append(i)
    CHMF_week_.append(i)
    CHMF_week_.append(i)
    CHMF_week_.append(i)
for i,k in zip(NLMK_week,week_counts_):
  if k == 4:
    NLMK_week_.append(i)
    NLMK_week_.append(i)
    NLMK_week_.append(i)
    NLMK_week_.append(i)
  elif k == 5:
    NLMK_week_.append(i)
    NLMK_week_.append(i)
    NLMK_week_.append(i)
    NLMK_week_.append(i)
    NLMK_week_.append(i)
for i,k in zip(MAGN_week,week_counts_):
  if k == 4:
    MAGN_week_.append(i)
    MAGN_week_.append(i)
    MAGN_week_.append(i)
    MAGN_week_.append(i)
  elif k == 5:
    MAGN_week_.append(i)
    MAGN_week_.append(i)
    MAGN_week_.append(i)
    MAGN_week_.append(i)
    MAGN_week_.append(i)

print(len(CHMF_week_))
print(len(NLMK_week_))
print(len(MAGN_week_))

gg = pd.DataFrame({'dt':date_list,'CHMF_week_':CHMF_week_,'NLMK_week_':NLMK_week_,'MAGN_week_':MAGN_week_})
print(gg)
