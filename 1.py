from __future__ import unicode_literals
from hazm import *

normalizer = Normalizer()
print (normalizer.normalize('اصلاح نويسه ها و استفاده از نیم‌فاصله پردازش را آسان مي كند'))
print (word_tokenize(('ما هم برای وصل کردن آمدیم! ولی برای پردازش، جدا بهتر نیست؟')))

__author__ = 'mehrdad'
import os
import sys
from operator import add
import numpy as np


try:
    from pyspark import SparkContext
    from pyspark import SparkConf

    sc = SparkContext(appName="PythonWordCount")
    lines =  sc.textFile('/home/mehrdad/farsi.txt', 1)


    counts = lines.flatMap(lambda x: word_tokenize(x)).map(lambda x: (x, 1)).reduceByKey(add)
    output = counts.collect()
    for (word, count) in output:
        print("%s: %i" % (normalizer.normalize(word), count))

    sc.stop()
except ImportError as e:
    print ("Can not import Spark Modules", e)



try:
    from pyspark import SparkContext
    from pyspark import SparkConf

    from pyspark.mllib.fpm import FPGrowth
    from pyspark import SparkContext
    from pyspark import SparkConf
    from pyspark import SparkContext
    from pyspark.conf import SparkConf

    conf = SparkConf()



    conf.setMaster("local").setAppName('FPgrowth-notebook').set("spark.executor.memory", "50g")

    sc = SparkContext(conf=conf)
    data = sc.textFile("/home/mehrdad/Downloads/Text.csv",10)
    transactions = data.map(lambda line: line.strip().split(' '))
    model = FPGrowth.train(transactions, minSupport=0.0001, numPartitions=10)
    result = model.freqItemsets().collect()
    f2=open("/home/mehrdad/Downloads/RulesOfBank.txt",'w')
    for fi in result:
        print(fi)
        f2.write(fi)
    sc.stop()
except ImportError as e:
    print ("Can not import Spark Modules", e)
