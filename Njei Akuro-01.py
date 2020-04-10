#!/usr/bin/env python
# coding: utf-8

# Printing the value count for pregnum (variable)
# It indicates how many times each respondent has been pregnant
# The output seems to be correct because the total of 13593 matches the total in the code book

# In[239]:


preg.pregnum.value_counts


# The Histogram shows that most pregnancies have a length of 39 weeks. 39 weeks can therefore be used to represent the whole population because it has the highest frequency.It implies that a pregnancy duration of 39 weeks is pretty much normal.Ignoring the part of the Histogram represennting other babies and focusing on first babies.39 weeks still has the tallest bar.This means that most first babies are born at 39 weeks.Therefore the assumption that first babies arrive late is false 

# In[285]:


width = 0.45
thinkplot.PrePlot(2)
thinkplot.Hist(first_hist, align='right', width=width)
thinkplot.Hist(other_hist, align='left', width=width)
thinkplot.Show(xlabel='weeks', ylabel='frequency',
xlim=[27, 46])


# The below code displays the pregancy length in weeks.For example 4744 pregnancies had a length of 39 weeks, 1120 pregnancies had length of 40 weeks.39 weeks has the highest value (mode) as also indicated on this Histogram above.
# 

# In[249]:


preg.prglngth.value_counts()


# I am not sure why the below code is giving mode to be 7. I believe the mode should be 39 weeks. I ran the test mode code in the solution section and it said the assertion error

# In[331]:


def Mode(hist):
    """Returns the value with the highest frequency.

    hist: Hist object

    returns: value from Hist
    """
    p, x = max([(p, x) for x, p in hist.Items()])
    return x
print(mode)


# In[333]:


mode = Mode(hist)
print('Mode of preg length', mode)
assert(mode == 39)


# To investigate whether first babies are lighter, we need a code that displays the weight of first babies, then we compare to the mean weight of 'not first babies (other babies).From the result of the code below, mean of first babies is 7.2 which is lesser than mean of other babies of 7.3.Therefore first babies are lighter than others

# In[326]:


mean0 = live.totalwgt_lb.mean()
mean1 = firsts.totalwgt_lb.mean()
mean2 = others.totalwgt_lb.mean()

print('Mean')
print('First babies', mean1)
print('Others', mean2)


# In[254]:


preg.totalwgt_lb.value_counts()[7.5] 


# In[255]:


preg.totalwgt_lb.value_counts()


# In[236]:


# Printing column names
preg = nsfg.ReadFemPreg()
preg.head()


# In[335]:


# Number of records with a 1st pregnancy count
preg.pregordr.value_counts()[1]


# In[336]:


preg.pregordr.value_counts()


# In[232]:


# A function that reads 2002FemResp.dat.gz
import numpy as np
import sys
import nsfg
import thinkstats2
def ReadFemResp(dct_file='2002FemResp.dct',
                dat_file='2002FemResp.dat.gz',
                nrows=None):
    
    dct = thinkstats2.ReadStataDct(dct_file)
    df = dct.ReadFixedWidth(dat_file, compression='gzip', nrows=nrows)
    #CleanFemResp(df)
    return df


# In[202]:


def CleanFemResp(df):
    """Recodes variables from the respondent frame.

    df: DataFrame
    """
    pass


def ValidatePregnum(resp):
    """Validate pregnum in the respondent file.

    resp: respondent DataFrame
    """
    # read the pregnancy frame
    preg = nsfg.ReadFemPreg()

    # make the map from caseid to list of pregnancy indices
    preg_map = nsfg.MakePregMap(preg)
    
    # iterate through the respondent pregnum series
    for index, pregnum in resp.pregnum.items():
        caseid = resp.caseid[index]
        indices = preg_map[caseid]

        # check that pregnum from the respondent file equals
        # the number of records in the pregnancy file
        if len(indices) != pregnum:
            print(caseid, len(indices), pregnum)
            return False

    return True


# In[212]:


def main(script):
    """Tests the functions in this module.

    script: string script name
    """
    resp = ReadFemResp()

    assert(len(resp) == 7643)
    assert(resp.pregnum.value_counts()[1] == 1267)
    assert(ValidatePregnum(resp))

    print('%s: All tests passed.' % script)


# In[204]:


"""This file contains code for use with "Think Stats",
by Allen B. Downey, available from greenteapress.com

Copyright 2010 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function, division

import sys
import numpy as np
import thinkstats2

from collections import defaultdict


def ReadFemResp(dct_file='2002FemResp.dct',
                dat_file='2002FemResp.dat.gz',
                nrows=None):
    """Reads the NSFG respondent data.

    dct_file: string file name
    dat_file: string file name

    returns: DataFrame
    """
    dct = thinkstats2.ReadStataDct(dct_file)
    df = dct.ReadFixedWidth(dat_file, compression='gzip', nrows=nrows)
    CleanFemResp(df)
    return df


def CleanFemResp(df):
    """Recodes variables from the respondent frame.

    df: DataFrame
    """
    pass


def ReadFemPreg(dct_file='2002FemPreg.dct',
                dat_file='2002FemPreg.dat.gz'):
    """Reads the NSFG pregnancy data.

    dct_file: string file name
    dat_file: string file name

    returns: DataFrame
    """
    dct = thinkstats2.ReadStataDct(dct_file)
    df = dct.ReadFixedWidth(dat_file, compression='gzip')
    CleanFemPreg(df)
    return df


def CleanFemPreg(df):
    """Recodes variables from the pregnancy frame.

    df: DataFrame
    """
    # mother's age is encoded in centiyears; convert to years
    df.agepreg /= 100.0

    # birthwgt_lb contains at least one bogus value (51 lbs)
    # replace with NaN
    df.loc[df.birthwgt_lb > 20, 'birthwgt_lb'] = np.nan
    
    # replace 'not ascertained', 'refused', 'don't know' with NaN
    na_vals = [97, 98, 99]
    df.birthwgt_lb.replace(na_vals, np.nan, inplace=True)
    df.birthwgt_oz.replace(na_vals, np.nan, inplace=True)
    df.hpagelb.replace(na_vals, np.nan, inplace=True)

    df.babysex.replace([7, 9], np.nan, inplace=True)
    df.nbrnaliv.replace([9], np.nan, inplace=True)

    # birthweight is stored in two columns, lbs and oz.
    # convert to a single column in lb
    # NOTE: creating a new column requires dictionary syntax,
    # not attribute assignment (like df.totalwgt_lb)
    df['totalwgt_lb'] = df.birthwgt_lb + df.birthwgt_oz / 16.0    

    # due to a bug in ReadStataDct, the last variable gets clipped;
    # so for now set it to NaN
    df.cmintvw = np.nan


def ValidatePregnum(resp, preg):
    """Validate pregnum in the respondent file.

    resp: respondent DataFrame
    preg: pregnancy DataFrame
    """
    # make the map from caseid to list of pregnancy indices
    preg_map = MakePregMap(preg)
    
    # iterate through the respondent pregnum series
    for index, pregnum in resp.pregnum.iteritems():
        caseid = resp.caseid[index]
        indices = preg_map[caseid]

        # check that pregnum from the respondent file equals
        # the number of records in the pregnancy file
        if len(indices) != pregnum:
            print(caseid, len(indices), pregnum)
            return False

    return True


def MakePregMap(df):
    """Make a map from caseid to list of preg indices.

    df: DataFrame

    returns: dict that maps from caseid to list of indices into `preg`
    """
    d = defaultdict(list)
    for index, caseid in df.caseid.iteritems():
        d[caseid].append(index)
    return d

def main():

    """Tests the functions in this module.

    script: string script name
    """
    # read and validate the respondent file
    resp = ReadFemResp()

    assert(len(resp) == 7643)
    assert(resp.pregnum.value_counts()[1] == 1267)

    # read and validate the pregnancy file
    preg = ReadFemPreg()
    print(preg.shape)

    assert len(preg) == 13593
    assert preg.caseid[13592] == 12571
    assert preg.pregordr.value_counts()[1] == 5033
    assert preg.nbrnaliv.value_counts()[1] == 8981
    assert preg.babysex.value_counts()[1] == 4641
    assert preg.birthwgt_lb.value_counts()[7] == 3049
    assert preg.birthwgt_oz.value_counts()[0] == 1037
    assert preg.prglngth.value_counts()[39] == 4744
    assert preg.outcome.value_counts()[1] == 9148
    assert preg.birthord.value_counts()[1] == 4413
    assert preg.agepreg.value_counts()[22.75] == 100
    assert preg.totalwgt_lb.value_counts()[7.5] == 302

    weights = preg.finalwgt.value_counts()
    key = max(weights.keys())
    assert preg.finalwgt.value_counts()[key] == 6

    # validate that the pregnum column in `resp` matches the number
    # of entries in `preg`
    assert(ValidatePregnum(resp, preg))

    
    print('All tests passed.')


if __name__ == '__main__':
    main()


# In[48]:


#Displaying dataframe
def main():
    resp = ReadFemResp()
    print(resp)
main()


# In[22]:


#printing column names
preg.columns


# In[21]:


#Selecting a single column name
preg.columns[1]


# In[23]:


#Selecting a column and check what type it is
pregordr = preg['pregordr']
type(pregordr)


# In[24]:


#Printing a column
pregordr


# In[26]:


#Selecting a single element from the column
pregordr[0]


# In[27]:


#Selecting a slice from a column
pregordr[2:5]


# In[28]:


#Selecting a column using dot notation
pregordr = preg.pregordr


# In[29]:


#Counting the number of times each value occurs
preg.outcome.value_counts().sort_index()


# In[159]:


df.loc[df.birthwgt_lb > 20, 'birthwgt_lb'] = np.nan


# In[165]:





# In[166]:


#Checking the values of another variable
preg.birthwgt_lb.value_counts().sort_index()


# In[31]:


#Making a dictionary that maps from each respondent's caseid 
#to a list of indices into the pregnancy DataFrame. 
#Use it to select the pregnancy outcomes for a single respondent.
caseid = 10229
preg_map = nsfg.MakePregMap(preg)
indices = preg_map[caseid]
preg.outcome[indices].values


# In[54]:


def main():
     # Tests the all the functions in this module.
    resp = ReadFemResp()
    assert(len(resp) == 7643)
    assert(resp.pregnum.value_counts()[1] == 1267)
    assert(ValidatePregnum(resp))
    print('All tests passed.')


# In[80]:


#Selecting the birthord column and printing the value count
#The total of 13593 is the same as the total provided in the codebook
birthord = preg['birthord']
type(birthord)
print(birthord)

#The first respondent had 1 child, the 2nd had 2 children, the 3rd had 1 child,....respondent 13592 had 3 children


# In[90]:


#Selecting the `prglngth` column and printing the value counts
prglngth = preg['prglngth']
type(prglngth)
print(prglngth)


# In[95]:


#computing mean of birthweight in pounds
preg.totalwgt_lb.mean()


# In[150]:


#Creating a new column totalwgt_kg
def CleanFemPreg(df):
    df.agepreg /= 100.0
na_vals = [97, 98, 99]
df.birthwgt_lb.replace(na_vals, np.nan, inplace=True)
df.birthwgt_oz.replace(na_vals, np.nan, inplace=True)
df['totalwgt_lb'] = df.birthwgt_lb + df.birthwgt_oz / 16
#The last line of this code above creates a new column totalwgt_lb


# In[ ]:





# In[145]:


df['totalwgt_lb'] = df.birthwgt_lb + df.birthwgt_oz / 16.0


# In[ ]:





# In[ ]:





# In[83]:


#Using isnull to count the number of nans.
preg.birthord.isnull().sum()


# In[69]:


# importing data frame
import nsfg
df = nsfg.ReadFemPreg()
df


# In[70]:


#Returning column names as unicode
df.columns


# In[151]:


df.columns[1]


# In[152]:


prglngth


# In[154]:


#counting number of times each value appears
df.outcome.value_counts().sort_index()


# In[156]:


#Displaying birthwight in lb and their counts e.g the count for baby weighing 15 pound is 1
df.birthwgt_lb.value_counts(sort=False)


# In[180]:


#because of the way the data files are organized, we do processing to get pregnancy data for each respondent
def MakePregMap(df):
    d = defaultdict(list)
    for index, caseid in df.caseid.iteritems():
        d[caseid].append(index)
    return d


# In[184]:


#code 4 is miscarriage, code 1 is life birth.This respondent had 6 miscarriages and 1 life birth
caseid = 10229
preg_map = nsfg.MakePregMap(df)
indices = preg_map[caseid]
df.outcome[indices].values


# In[185]:


preg = nsfg.ReadFemPreg()


# In[190]:


if len(indices) != pregnum:
           print(caseid, len(indices), pregnum)
           return False

           return True


# In[199]:


# read the pregnancy frame
preg = nsfg.ReadFemPreg()

   # make the map from caseid to list of pregnancy indices
preg_map = nsfg.MakePregMap(preg)
   
   # iterate through the respondent pregnum series
   for index, pregnum in resp.pregnum.items():
       caseid = resp.caseid[index]
       indices = preg_map[caseid]


# In[223]:


#Printing the value count for birthord
preg.birthord.value_counts


# In[217]:


# read and validate pregnancy file
preg = ReadFemPreg()
print(preg.shape)


# In[222]:


#Printing the value count for pregnum
#It indicates howmany times each respondent has been pregnant
preg.pregnum.value_counts


# In[225]:


df.agepreg /= 100.0


# In[228]:


#list the ages of the respondents
preg.agepreg.value_counts


# In[262]:


import thinkstats2
hist = thinkstats2.Hist([1, 2, 2, 3, 5])
hist


# In[272]:


hist.Freq(2)


# In[273]:


hist.Values()



# In[275]:


for val in sorted(hist.Values()):
    print(val, hist.Freq(val))


# In[277]:


for val, freq in hist.Items():
    print(val, freq)


# In[278]:


import thinkplot
thinkplot.Hist(hist)
thinkplot.Show(xlabel='value', ylabel='frequency')


# In[279]:


preg = nsfg.ReadFemPreg()
live = preg[preg.outcome == 1]


# In[280]:


hist = thinkstats2.Hist(live.birthwgt_lb, label='birthwgt_lb')
thinkplot.Hist(hist)
thinkplot.Show(xlabel='pounds', ylabel='frequency')


# In[282]:


for weeks, freq in hist.Smallest(10):
    print(weeks, freq)


# In[283]:


firsts = live[live.birthord == 1]
others = live[live.birthord != 1]
first_hist = thinkstats2.Hist(firsts.prglngth, label='first')
other_hist = thinkstats2.Hist(others.prglngth, label='other')


# In[284]:


width = 0.45
thinkplot.PrePlot(2)
thinkplot.Hist(first_hist, align='right', width=width)
thinkplot.Hist(other_hist, align='left', width=width)
thinkplot.Show(xlabel='weeks', ylabel='frequency',
xlim=[27, 46])


# In[332]:


mode = Mode(hist)
print('Mode of preg length', mode)
assert(mode == 39)


# In[307]:


modes = AllModes(hist)
assert(modes[0][1] == 4693)
for value, freq in modes[:5]:
    print(value, freq)


# In[305]:


from __future__ import print_function

import sys
from operator import itemgetter

import first
import thinkstats2


def Mode(hist):
    """Returns the value with the highest frequency.

    hist: Hist object

    returns: value from Hist
    """
    p, x = max([(p, x) for x, p in hist.Items()])
    return x


# In[304]:



def AllModes(hist):
    """Returns value-freq pairs in decreasing order of frequency.

    hist: Hist object

    returns: iterator of value-freq pairs
    """
    return sorted(hist.Items(), key=itemgetter(1), reverse=True)


# In[310]:



def WeightDifference(live, firsts, others):
    """Explore the difference in weight between first babies and others.

    live: DataFrame of all live births
    firsts: DataFrame of first babies
    others: DataFrame of others
    """
    mean0 = live.totalwgt_lb.mean()
    mean1 = firsts.totalwgt_lb.mean()
    mean2 = others.totalwgt_lb.mean()

    var1 = firsts.totalwgt_lb.var()
    var2 = others.totalwgt_lb.var()

    print('Mean')
    print('First babies', mean1)
    print('Others', mean2)


# In[317]:


def WeightDifference(live, firsts, others):
    """Explore the difference in weight between first babies and others.

    live: DataFrame of all live births
    firsts: DataFrame of first babies
    others: DataFrame of others
    """
    mean0 = live.totalwgt_lb.mean()
    mean1 = firsts.totalwgt_lb.mean()
    mean2 = others.totalwgt_lb.mean()

    var1 = firsts.totalwgt_lb.var()
    var2 = others.totalwgt_lb.var()

    print('Mean')
    print('First babies', mean1)
    print('Others', mean2)

    print('Variance')
    print('First babies', var1)
    print('Others', var2)

    print('Difference in lbs', mean1 - mean2)
    print('Difference in oz', (mean1 - mean2) * 16)

    print('Difference relative to mean (%age points)', 
          (mean1 - mean2) / mean0 * 100)

    d = thinkstats2.CohenEffectSize(firsts.totalwgt_lb, others.totalwgt_lb)
    print('Cohen d', d)


# In[319]:


d = thinkstats2.CohenEffectSize(firsts.totalwgt_lb, others.totalwgt_lb)
print('Cohen d', d)


# In[325]:


mean0 = live.totalwgt_lb.mean()
mean1 = firsts.totalwgt_lb.mean()
mean2 = others.totalwgt_lb.mean()

print('Mean')
print('First babies', mean1)
print('Others', mean2)


# In[330]:


def Mode(hist):
    """Returns the value with the highest frequency.

    hist: Hist object

    returns: value from Hist
    """
    p, x = max([(p, x) for x, p in hist.Items()])
    return x
print(mode)

