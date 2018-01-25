# -*- coding: utf-8 -*-

import numpy as np
import pickle

from pyspark import SparkContext
import itertools
from scipy.sparse import hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.pipeline import Pipeline
from spark_sklearn import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt

def filelister(filename):  # Bssed on: http://stackoverflow.com/questions/6375343/reading-unicode-elements-into-numpy-array
    return pickle.load(open(filename, "rb"))

def dataset_helper(fns,gender_choice='m',text_choice=False,pron_choice=False,pron_tag_choice=False, pred_choice=False,pred_tag_choice=False, lix_choice=False, heylighen_fmeasure_choice=False):
    features = []
    feature_indices={}
    category = 3

    if gender_choice=='m':
        gender_num=1
    else:
        gender_num=2

    for fn in fns:
        f = filelister(fn)
        if text_choice:
            for f_name, f_text in f:
                features.append((f_text, gender_num, category))
            category += 1

        else:
            for author_forename, tag_list in f:
                content_words = ''

                pron_nums = 0
                pred_nums = 0

                word_count = 0
                long_word_count = 0
                sent_count = 0
                lix_num = 0

                noun_count = 0
                adj_count = 0
                adp_count = 0
                det_count = 0

                pron_count = 0
                verb_count = 0
                adv_count = 0
                intj_count = 0

                feature_row = []
                for word, tag in tag_list:
                    if tag == "ADJ" or tag == "ADV" or tag == "INTJ" or tag == "NOUN" or tag == "PUNCT" or tag == "VERB":
                        if tag != "PUNCT":
                            content_words += ' ' + word
                        elif tag == "PUNCT" and word == '.' or word == ',' or word == '!' or word == '?':
                            content_words += word
                            sent_count += 1
                    if pron_choice:
                        if tag == "PRON" and word.lower() == 'jeg' or word.lower() == 'mig' or word.lower() == 'vi' or word.lower() == 'os':
                            pron_nums += 1
                    if pron_tag_choice:
                        if tag == "PRON" and word.lower() == 'jeg' or word.lower() == 'mig' or word.lower() == 'vi' or word.lower() == 'os':
                            content_words += ' ' + word
                    if pred_choice:
                        if tag == "ADP":
                            pred_nums += 1
                    if pred_tag_choice:
                        if tag == "ADP":
                            content_words += ' ' + word
                    if lix_choice:
                        word_count += 1
                        if len(word) > 6:
                            long_word_count += 1
                    if heylighen_fmeasure_choice:
                        if tag == "NOUN":
                            noun_count += 1
                        elif tag == "ADJ":
                            adj_count += 1
                        elif tag == "ADP":
                            adp_count += 1
                        elif tag == "DET":
                            det_count += 1

                        elif tag == "PRON":
                            pron_count += 1
                        elif tag == "VERB":
                            verb_count += 1
                        elif tag == "ADV":
                            adv_count += 1
                        elif tag == "INTJ":
                            intj_count += 1

                feature_row.append(content_words)
                feature_row.append(gender_num)
                feature_row.append(category)

                if pron_choice:
                    feature_row.append(pron_nums)
                    feature_indices["pron"]=feature_row.index(pron_nums)
                if pred_choice:
                    feature_row.append(pred_nums)
                    feature_indices["pred"] = feature_row.index(pred_nums)
                if lix_choice:
                    try:
                        lix_one = word_count / sent_count
                        lix_two = (long_word_count * 100) / word_count
                        lix_num += lix_one + lix_two
                    except ZeroDivisionError:
                        lix_num = 0
                    feature_row.append(lix_num)
                    feature_indices["lix"] = feature_row.index(lix_num)
                if heylighen_fmeasure_choice:
                    heylighen_fmeasure = .5 * ((noun_count + adj_count + adp_count + det_count) -
                                               (pron_count + verb_count + adv_count + intj_count) + 100)
                    feature_row.append(heylighen_fmeasure)
                    feature_indices["heylighen"] = feature_row.index(heylighen_fmeasure)

                features.append(feature_row)

            category += 1

    return np.array(features),feature_indices

def dataset_trimmer(dataset_boys,dataset_girls):
    if len(dataset_girls) > len(dataset_boys):
        dataset_girls = dataset_girls[:len(dataset_boys)]
    elif len(dataset_girls) < len(dataset_boys):
        dataset_boys = dataset_boys[:len(dataset_girls)]
    print(len(dataset_boys))
    print(len(dataset_girls))
    return dataset_boys,dataset_girls


def stats(dataset, fn):
    f = open(fn, "w")
    print(dataset[:, 2])
    boys = dataset.copy()[np.where(dataset[:, 1] == '1')]
    girls = dataset.copy()[np.where(dataset[:, 1] == '2')]

    berlingske = dataset.copy()[np.where(dataset[:, 2] == '3')]
    business = dataset.copy()[np.where(dataset[:, 2] == '4')]
    food = dataset.copy()[np.where(dataset[:, 2] == '5')]
    travel = dataset.copy()[np.where(dataset[:, 2] == '6')]
    jv = dataset.copy()[np.where(dataset[:, 2] == '7')]

    boys_num = len(boys)
    girls_num = len(girls)

    boys_word_num = [len(text.split()) for text in boys[:, 0]]
    girls_word_num = [len(text.split()) for text in girls[:, 0]]

    boys_word_num_avg = np.mean(np.array(boys_word_num))
    girls_word_num_avg = np.mean(np.array(girls_word_num))

    berlingske_num = len(berlingske)
    business_num = len(business)
    food_num = len(food)
    travel_num = len(travel)
    jv_num = len(jv)

    berlingske_word_num = [len(text.split()) for text in berlingske[:, 0]]
    business_word_num = [len(text.split()) for text in business[:, 0]]
    food_word_num = [len(text.split()) for text in food[:, 0]]
    travel_word_num = [len(text.split()) for text in travel[:, 0]]
    jv_word_num = [len(text.split()) for text in jv[:, 0]]

    berlingske_word_num_avg = np.mean(np.array(berlingske_word_num))
    business_word_num_avg = np.mean(np.array(business_word_num))
    food_word_num_avg = np.mean(np.array(food_word_num))
    travel_word_num_avg = np.mean(np.array(travel_word_num))
    jv_word_num_avg = np.mean(np.array(jv_word_num))

    f.write("Gender/Category,Number of texts,Average number of words pr. text\n")
    f.write("Male," + str(boys_num) + "," + str(boys_word_num_avg) + "\n")
    f.write("Female," + str(girls_num) + "," + str(girls_word_num_avg) + "\n")
    f.write("Berlingske," + str(berlingske_num) + "," + str(berlingske_word_num_avg) + "\n")
    f.write("Business," + str(business_num) + "," + str(business_word_num_avg) + "\n")
    f.write("Food," + str(food_num) + "," + str(food_word_num_avg) + "\n")
    f.write("Travel," + str(travel_num) + "," + str(travel_word_num_avg) + "\n")
    f.write("JydskeVestkysten," + str(jv_num) + "," + str(jv_word_num_avg))

    f.close()

class NumFeatureAdder(BaseEstimator, TransformerMixin):

    def __init__(self, nums_train,nums_test):
        self.nums_train = nums_train
        self.nums_test = nums_test


    def transform(self, X, y=None):
        if np.any(np.not_equal(self.nums_train,None)) and np.any(np.not_equal(self.nums_test,None)):
            if len(self.nums_train)==X.shape[0]:
                X = hstack((X, self.nums_train))
            elif len(self.nums_test)==X.shape[0]:
                X = hstack((X, self.nums_test))
        return X

    def fit(self, X, y=None):
        return self

def dummy_clf(X_train, y_train, X_test,strategy="most_frequent",tfidf_choice=False, nums_train=None,nums_test=None):
    if type(nums_train) is np.ndarray and type(nums_test) is np.ndarray:
        if tfidf_choice:
            clf_pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1, 10))),
                            ('tfidf', TfidfTransformer(smooth_idf=False)),
                            ('numfeat', NumFeatureAdder(nums_train,nums_test)),
                            ('clf',DummyClassifier(strategy=strategy, random_state=None, constant=None))])
        else:
            clf_pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1, 10))),
            ('numfeat', NumFeatureAdder(nums_train, nums_test)),
            ('clf',DummyClassifier(strategy=strategy, random_state=None, constant=None))])
    else:
        if tfidf_choice:
            clf_pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1, 10))),
                                     ('tfidf', TfidfTransformer(smooth_idf=False)),
                                     ('clf',DummyClassifier(strategy=strategy, random_state=None, constant=None))])
        else:
            clf_pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1, 10))),
                                     ('clf',DummyClassifier(strategy=strategy, random_state=None, constant=None))])

    clf_dummy = clf_pipeline.fit(X_train,y_train)
    dummy_predicted = clf_dummy.predict(X_test)

    return dummy_predicted

def multinom_clf(X_train, y_train, X_test,tfidf_choice=False, nums_train=None,nums_test=None):
    if type(nums_train) is np.ndarray and type(nums_test) is np.ndarray:
        if tfidf_choice:
            clf_pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1, 10))),
                            ('tfidf', TfidfTransformer(smooth_idf=False)),
                            ('numfeat', NumFeatureAdder(nums_train,nums_test)),
                            ('clf',MultinomialNB())])
        else:
            clf_pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1, 10))),
            ('numfeat', NumFeatureAdder(nums_train, nums_test)),
            ('clf',MultinomialNB())])
    else:
        if tfidf_choice:
            clf_pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1, 10))),
                                     ('tfidf', TfidfTransformer(smooth_idf=False)),
                                     ('clf',MultinomialNB())])
        else:
            clf_pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1, 10))),
                                     ('clf',MultinomialNB())])

    clf_multinom = clf_pipeline.fit(X_train, y_train)
    multinomial_predicted = clf_multinom.predict(X_test)

    return multinomial_predicted


def sgd_clf(X_train, y_train, X_test,tfidf_choice=False,nums_train=None,nums_test=None):
    if type(nums_train) is np.ndarray and type(nums_test) is np.ndarray:
        if tfidf_choice:
            clf_pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1, 10))),
                            ('tfidf', TfidfTransformer(smooth_idf=False)),
                            ('numfeat', NumFeatureAdder(nums_train,nums_test)),
                            ('clf',SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=5, random_state=42))])
        else:
            clf_pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1, 10))),
                                     ('numfeat', NumFeatureAdder(nums_train, nums_test)),
                                     ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, max_iter=5, random_state=42))])
    else:
        if tfidf_choice:
            clf_pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1, 10))),
                                     ('tfidf', TfidfTransformer(smooth_idf=False)),
                                     ('clf',SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, max_iter=5, random_state=42))])
        else:
            clf_pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1, 10))),
                                     ('clf',SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, max_iter=5, random_state=42))])
    clf_sgd = clf_pipeline.fit(X_train, y_train)
    sgd_predicted = clf_sgd.predict(X_test)

    return sgd_predicted

def grid_search_svm(X_train, y_train,X_test,ngrams,n_split,svm_choice='linear',tfidf_choice=False,nums_train=None,nums_test=None):
    svm=None
    grid=None

    if svm_choice == 'linear':
        svm = LinearSVC()
        c_array = np.logspace(1., 4., num=4)
        if tfidf_choice:
            grid = {'vect__ngram_range': ngrams, 'tfidf__use_idf': (True, False),
                       'clf__C': c_array.tolist()}
        else:
            grid = {'vect__ngram_range': ngrams,
                    'clf__C': c_array.tolist()}

    elif svm_choice == 'svc':
        svm = SVC()
        c_array = np.logspace(-3., 6., num=10)
        g_array = np.logspace(-3., 3., num=7)
        if tfidf_choice:
            grid = {'vect__ngram_range': ngrams,
                'tfidf__use_idf': (True, False),
                'clf__kernel': ['rbf'],
                'clf__C': c_array.tolist(),
                'clf__gamma': g_array.tolist()}
        else:
            grid = {'vect__ngram_range': ngrams,
                    'clf__kernel': ['rbf'],
                    'clf__C': c_array.tolist(),
                    'clf__gamma': g_array.tolist()}

    if type(nums_train) is np.ndarray and type(nums_test) is np.ndarray:
        if tfidf_choice:
            clf_pipeline = Pipeline([('vect', CountVectorizer(ngram_range=ngrams)),
                                     ('tfidf', TfidfTransformer(smooth_idf=False)),
                                     ('numfeat', NumFeatureAdder(nums_train,nums_test)),
                                     ('clf',svm)])
        else:
            clf_pipeline = Pipeline([('vect', CountVectorizer(ngram_range=ngrams)),
                                     ('numfeat', NumFeatureAdder(nums_train, nums_test)),
                                     ('clf', svm)])
    else:
        if tfidf_choice:
            clf_pipeline = Pipeline([('vect', CountVectorizer(ngram_range=ngrams)),
                                     ('tfidf', TfidfTransformer(smooth_idf=False)),
                                     ('clf',svm)])
        else:
            clf_pipeline = Pipeline([('vect', CountVectorizer(ngram_range=ngrams)),
                                     ('clf',svm)])
    print(clf_pipeline.get_params().keys())

    sc = SparkContext.getOrCreate()
    grid_search = GridSearchCV(sc, clf_pipeline, grid, n_jobs=-1, cv=n_split)
    grid_search.fit(X_train, y_train)
    grid_search_predicted = grid_search.predict(X_test)

    return grid_search_predicted


def plot_confusion_matrix(cm, classes,title='Confusion matrix - Gender and Categories',cmap=plt.cm.Reds):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def gender_clf(fns_boys, fns_girls, fn_out,fn_out_multinom,fn_out_sgd,fn_out_linsvc,fn_out_svc,fn_out_dummy,idx_out=None,train_out=None,test_out=None,indices=None,train_indices=None,test_indices=None,dataset_num=None,text_choice=False,tfidf_choice=False,pron_choice=False,pron_tag_choice=False,pred_choice=False,pred_tag_choice=False,lix_choice=False,heylighen_fmeasure_choice=False):
    global lix_train
    dataset_boys,feature_indices = dataset_helper(fns_boys,text_choice=text_choice, pron_choice=pron_choice,pron_tag_choice=pron_tag_choice,pred_choice=pred_choice,pred_tag_choice=pred_tag_choice,lix_choice=lix_choice,heylighen_fmeasure_choice=heylighen_fmeasure_choice)
    dataset_girls, _ = dataset_helper(fns_girls,gender_choice='f',text_choice=text_choice, pron_choice=pron_choice,pron_tag_choice=pron_tag_choice,pred_choice=pred_choice,pred_tag_choice=pred_tag_choice,lix_choice=lix_choice,heylighen_fmeasure_choice=heylighen_fmeasure_choice)

    dataset = np.append(dataset_boys, dataset_girls, axis=0)

    if text_choice:
        stats(dataset, "stats_text_org.txt")
    else:
        stats(dataset, "stats_pos_org.txt")

    dataset_boys_berlingske,dataset_girls_berlingske=dataset_trimmer(dataset_boys.copy()[dataset_boys[:, 2]== "3"],dataset_girls.copy()[dataset_girls[:, 2]== "3"])
    dataset_boys_business, dataset_girls_business = dataset_trimmer(dataset_boys.copy()[dataset_boys[:, 2] == "4"], dataset_girls.copy()[dataset_girls[:, 2] == "4"])
    dataset_boys_food, dataset_girls_food = dataset_trimmer(dataset_boys.copy()[dataset_boys[:, 2] == "5"], dataset_girls.copy()[dataset_girls[:, 2] == "5"])
    dataset_boys_travel, dataset_girls_travel = dataset_trimmer(dataset_boys.copy()[dataset_boys[:, 2] == "6"], dataset_girls.copy()[dataset_girls[:, 2] == "6"])
    dataset_boys_jv, dataset_girls_jv = dataset_trimmer(dataset_boys.copy()[dataset_boys[:, 2] == "7"], dataset_girls.copy()[dataset_girls[:, 2] == "7"])

    dataset_boys_new=np.concatenate((dataset_boys_berlingske,dataset_boys_business,dataset_boys_food,dataset_boys_travel,dataset_boys_jv),axis=0)
    dataset_girls_new=np.concatenate((dataset_girls_berlingske,dataset_girls_business,dataset_girls_food,dataset_girls_travel,dataset_girls_jv),axis=0)


    dataset = np.append(dataset_boys_new, dataset_girls_new, axis=0)

    if text_choice:
        stats(dataset, "stats_text.txt")
    else:
        stats(dataset,"stats_pos.txt")


    boys_labels = dataset_boys_new.copy()[:, 1]
    girls_labels = dataset_girls_new.copy()[:, 1]

    if indices is None:
        indices = np.random.permutation(dataset_boys_new.shape[0])

    indices_train, indices_test = indices[:(len(dataset_boys_new)//5)*3],indices[(len(dataset_boys_new)//5)*3:]

    boys_train,boys_test,boys_labels_train,boys_labels_test,boys_features_train,boys_features_test = dataset_boys_new.copy()[:,0][indices_train],dataset_boys_new.copy()[:,0][indices_test],boys_labels[indices_train],boys_labels[indices_test], dataset_boys_new.copy()[:, 3:][indices_train], dataset_boys_new.copy()[:, 3:][indices_test]
    girls_train, girls_test, girls_labels_train, girls_labels_test,girls_features_train,girls_features_test = dataset_girls_new.copy()[:, 0][indices_train],dataset_girls_new.copy()[:, 0][indices_test],girls_labels[indices_train], girls_labels[indices_test],dataset_girls_new.copy()[:, 3:][indices_train], dataset_girls_new.copy()[:, 3:][indices_test]

    if train_indices is None:
        train_indices = np.random.permutation(len(boys_labels_train)*2)

    if test_indices is None:
        test_indices = np.random.permutation(len(boys_labels_test)*2)

    X_train = np.concatenate((boys_train.copy(),girls_train.copy()),axis=0)[train_indices]
    X_test = np.concatenate((boys_test.copy(), girls_test.copy()), axis=0)[test_indices]

    Y_train = np.concatenate((boys_labels_train.copy(), girls_labels_train.copy()), axis=0)[train_indices]
    Y_test = np.concatenate((boys_labels_test.copy(), girls_labels_test.copy()), axis=0)[test_indices]

    features_train = np.concatenate((boys_features_train.copy(), girls_features_train.copy()), axis=0)[train_indices]
    features_test = np.concatenate((boys_features_test.copy(), girls_features_test.copy()), axis=0)[test_indices]

    ngrams = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10)]

    if features_train.shape[1]>0 and features_test.shape[1]>0:
        dummy_predicted = dummy_clf(X_train, Y_train, X_test, tfidf_choice=tfidf_choice, nums_train=None, nums_test=None)
        multinom_predicted = multinom_clf(X_train, Y_train, X_test, tfidf_choice=tfidf_choice,
                                          nums_train=features_train.astype(np.float64), nums_test=features_test.astype(np.float64))
        sgd_predicted = sgd_clf(X_train, Y_train, X_test, tfidf_choice=tfidf_choice,
                                          nums_train=features_train.astype(np.float64), nums_test=features_test.astype(np.float64))
        linear_gs_predicted = grid_search_svm(X_train, Y_train, X_test, ngrams, 2, tfidf_choice=tfidf_choice,
                                          nums_train=features_train.astype(np.float64), nums_test=features_test.astype(np.float64))
        gs_predicted = grid_search_svm(X_train, Y_train, X_test, ngrams, 2, svm_choice='svc',tfidf_choice=tfidf_choice,
                                          nums_train=features_train.astype(np.float64), nums_test=features_test.astype(np.float64))

    else:
        dummy_predicted = dummy_clf(X_train, Y_train, X_test, tfidf_choice=tfidf_choice)
        multinom_predicted = multinom_clf(X_train, Y_train, X_test, tfidf_choice=tfidf_choice)
        sgd_predicted = sgd_clf(X_train, Y_train, X_test, tfidf_choice=tfidf_choice)
        linear_gs_predicted = grid_search_svm(X_train, Y_train, X_test, ngrams, 2, tfidf_choice=tfidf_choice)
        gs_predicted = grid_search_svm(X_train, Y_train, X_test, ngrams, 2, svm_choice='svc', tfidf_choice=tfidf_choice)


    f_out_dummy = open(fn_out_dummy, 'a')
    prec, rec, f1s, supp = precision_recall_fscore_support(Y_test, dummy_predicted)
    f_out_dummy.write(str(sum(prec) / 2) + "," + str(sum(rec) / 2) + "," + str(sum(f1s / 2)) + "\n")
    f_out_dummy.close()

    f_out_multinom = open(fn_out_multinom, 'a')
    prec,rec,f1s,supp = precision_recall_fscore_support(Y_test, multinom_predicted)
    f_out_multinom.write(str(sum(prec)/2)+","+str(sum(rec)/2)+","+str(sum(f1s/2))+"\n")
    f_out_multinom.close()

    f_out_sgd = open(fn_out_sgd, 'a')
    prec, rec, f1s, supp = precision_recall_fscore_support(Y_test, sgd_predicted)
    f_out_sgd.write(str(sum(prec)/2)+","+str(sum(rec)/2)+","+str(sum(f1s/2))+"\n")
    f_out_sgd.close()

    f_out_linsvc = open(fn_out_linsvc, 'a')
    prec, rec, f1s, supp = precision_recall_fscore_support(Y_test, linear_gs_predicted)
    f_out_linsvc.write(str(sum(prec)/2)+","+str(sum(rec)/2)+","+str(sum(f1s/2))+"\n")
    f_out_linsvc.close()

    f_out_svc = open(fn_out_svc, 'a')
    prec, rec, f1s, supp = precision_recall_fscore_support(Y_test, gs_predicted)
    f_out_svc.write(str(sum(prec)/2)+","+str(sum(rec)/2)+","+str(sum(f1s/2))+"\n")
    f_out_svc.close()

    class_names_gender = ['Male', 'Female']

    cf_matrix_dummy = confusion_matrix(Y_test, dummy_predicted)
    cf_matrix_multinom = confusion_matrix(Y_test, multinom_predicted)
    cf_matrix_sgd = confusion_matrix(Y_test, sgd_predicted)
    cf_matrix_linearsvc = confusion_matrix(Y_test, linear_gs_predicted)
    cf_matrix_svc = confusion_matrix(Y_test, gs_predicted)

    fn_out_raw = fn_out.split(".")[0]

    fn_out_dummy_pred = fn_out_raw + "_dummy.p"
    fn_out_multinom_pred = fn_out_raw + "_multinom.p"
    fn_out_sgd_pred = fn_out_raw + "_sgd.p"
    fn_out_linear_svc_pred = fn_out_raw + "_linear_svc.p"
    fn_out_svc_pred = fn_out_raw + "_svc.p"

    f_out_dummy = open(fn_out_dummy_pred,"wb")
    f_out_multinom = open(fn_out_multinom_pred,"wb")
    f_out_sgd = open(fn_out_sgd_pred,"wb")
    f_out_linear_svc = open(fn_out_linear_svc_pred,"wb")
    f_out_svc = open(fn_out_svc_pred,"wb")

    pickle.dump(dummy_predicted,f_out_dummy)
    f_out_dummy.close()
    pickle.dump(multinom_predicted, f_out_multinom)
    f_out_multinom.close()
    pickle.dump(sgd_predicted, f_out_sgd)
    f_out_sgd.close()
    pickle.dump(linear_gs_predicted, f_out_linear_svc)
    f_out_linear_svc.close()
    pickle.dump(gs_predicted, f_out_svc)
    f_out_svc.close()

    fn_out_dummy_cm = fn_out_raw + "_dummy.png"
    fn_out_multinom_cm=fn_out_raw+"_multinom.png"
    fn_out_sgd_cm = fn_out_raw+ "_sgd.png"
    fn_out_linear_svc_cm = fn_out_raw + "_linear_svc.png"
    fn_out_svc_cm = fn_out_raw + "_svc.png"

    plt.figure(1)
    plot_confusion_matrix(cf_matrix_dummy, classes=class_names_gender, title="Multinomial Clf - Dataset " + dataset_num)
    plt.savefig(fn_out_dummy_cm)

    plt.figure(2)
    plot_confusion_matrix(cf_matrix_multinom, classes=class_names_gender, title="Multinomial Clf - Dataset " + dataset_num)
    plt.savefig(fn_out_multinom_cm)

    plt.figure(3)
    plot_confusion_matrix(cf_matrix_sgd, classes=class_names_gender, title="SGD Clf - Dataset " + dataset_num)
    plt.savefig(fn_out_sgd_cm)

    plt.figure(4)
    plot_confusion_matrix(cf_matrix_linearsvc, classes=class_names_gender, title="Linear SVC Clf - Dataset " + dataset_num)
    plt.savefig(fn_out_linear_svc_cm)

    plt.figure(5)
    plot_confusion_matrix(cf_matrix_svc, classes=class_names_gender, title="SVC Clf - Dataset " + dataset_num)
    plt.savefig(fn_out_svc_cm)

    plt.close('all') # Based on https://stackoverflow.com/questions/17106288/matplotlib-pyplot-will-not-forget-previous-plots-how-can-i-flush-refresh

    if idx_out is not None:
        indices_out=open(idx_out,'wb')
        pickle.dump(indices,indices_out)
        indices_out.close()

    if train_out is not None:
        train_indices_out=open(train_out,'wb')
        pickle.dump(train_indices,train_indices_out)
        train_indices_out.close()

    if test_out is not None:
        test_indices_out=open(test_out,'wb')
        pickle.dump(test_indices,test_indices_out)
        test_indices_out.close()

    return indices,train_indices,test_indices


if __name__ == "__main__":
    filenames_boys=['text_berlingske_boys.p','text_business_boys.p','text_foodblogs_boys.p','text_travelblogs_boys.p','text_jv_boys.p']
    filenames_girls=['text_berlingske_girls.p','text_business_girls.p','text_foodblogs_girls.p','text_travelblogs_girls.p','text_jv_girls.p']
    idxs=filelister('idx_text.p')
    train_idxs=filelister('train_idx_text.p')
    test_idxs=filelister('test_idx_text.p')
    gender_clf(filenames_boys,filenames_girls,"svm_0_text_baseline_multilabel.txt",'score_multinom_text_org.txt','score_sgd_text_org.txt','score_linear_svc_text_org.txt','score_svc_text_org.txt','score_majority_text_org.txt',indices=idxs,train_indices=train_idxs,test_indices=test_idxs,dataset_num="1",text_choice=True)
    gender_clf(filenames_boys, filenames_girls, "svm_1_text_multilabel.txt",'score_multinom_text.txt','score_sgd_text.txt','score_linear_svc_text.txt','score_svc_text.txt','score_majority_text.txt',indices=idxs,train_indices=train_idxs,test_indices=test_idxs,dataset_num="1a",text_choice=True,tfidf_choice=True)
    filenames_boys = ['pos_berlingske_boys.p', 'pos_business_boys.p', 'pos_foodblogs_boys.p', 'pos_travelblogs_boys.p',
                        'pos_jv_boys.p']
    filenames_girls = ['pos_berlingske_girls.p', 'pos_business_girls.p', 'pos_foodblogs_girls.p',
                        'pos_travelblogs_girls.p', 'pos_jv_girls.p']
    idxs = filelister('idx_pos.p')
    train_idxs = filelister('train_idx_pos.p')
    test_idxs = filelister('test_idx_pos.p')
    gender_clf(filenames_boys, filenames_girls, 'svm_2_pos_pron_tag_multilabel_original.txt','score_multinom_pos_org.txt','score_sgd_pos_org.txt','score_linear_svc_pos_org.txt','score_svc_pos_org.txt','score_majority_pos_org.txt',indices=idxs,train_indices=train_idxs,test_indices=test_idxs,dataset_num="2",tfidf_choice=False)
    gender_clf(filenames_boys, filenames_girls,  'svm_3_pos_pron_tag_multilabel_original.txt','score_multinom_pos_org.txt','score_sgd_pos_org.txt','score_linear_svc_pos_org.txt','score_svc_pos_org.txt','score_majority_pos_org.txt',indices=idxs,train_indices=train_idxs,test_indices=test_idxs,dataset_num="3", pron_choice=True,tfidf_choice=False)
    gender_clf(filenames_boys, filenames_girls,  'svm_4_pos_pron_tag_multilabel_original.txt','score_multinom_pos_org.txt','score_sgd_pos_org.txt','score_linear_svc_pos_org.txt','score_svc_pos_org.txt','score_majority_pos_org.txt', indices=idxs,train_indices=train_idxs,test_indices=test_idxs,dataset_num="4", lix_choice=True,tfidf_choice=False)
    gender_clf(filenames_boys, filenames_girls,  'svm_5_pos_pron_tag_multilabel_original.txt','score_multinom_pos_org.txt','score_sgd_pos_org.txt','score_linear_svc_pos_org.txt','score_svc_pos_org.txt','score_majority_pos_org.txt',indices=idxs,train_indices=train_idxs,test_indices=test_idxs, dataset_num="5", heylighen_fmeasure_choice=True,tfidf_choice=False)

    gender_clf(filenames_boys, filenames_girls,  'svm_6_pos_pron_tag_multilabel_original.txt','score_multinom_pos_org.txt','score_sgd_pos_org.txt','score_linear_svc_pos_org.txt','score_svc_pos_org.txt','score_majority_pos_org.txt',indices=idxs,train_indices=train_idxs,test_indices=test_idxs, dataset_num="6", pred_choice=True,tfidf_choice=False)
    gender_clf(filenames_boys, filenames_girls,  'svm_7_pos_pron_tag_multilabel_original.txt', 'score_multinom_pos_org.txt','score_sgd_pos_org.txt','score_linear_svc_pos_org.txt','score_svc_pos_org.txt','score_majority_pos_org.txt',indices=idxs,train_indices=train_idxs,test_indices=test_idxs,dataset_num="7", pron_tag_choice=True,tfidf_choice=False)
    gender_clf(filenames_boys, filenames_girls, 'svm_8_pos_pred_tag_multilabel_original.txt', 'score_multinom_pos_org.txt','score_sgd_pos_org.txt','score_linear_svc_pos_org.txt','score_svc_pos_org.txt','score_majority_pos_org.txt',indices=idxs,train_indices=train_idxs,test_indices=test_idxs, dataset_num="8", pred_tag_choice=True,tfidf_choice=False)

    gender_clf(filenames_boys, filenames_girls,  'svm_9_pos_pron_pred_multilabel_original.txt', 'score_multinom_pos_org.txt','score_sgd_pos_org.txt','score_linear_svc_pos_org.txt','score_svc_pos_org.txt','score_majority_pos_org.txt',indices=idxs,train_indices=train_idxs,test_indices=test_idxs,dataset_num="9",pron_choice=True,pred_choice=True,tfidf_choice=False)
    gender_clf(filenames_boys, filenames_girls,  'svm_10_pos_pron_tag_pred_tag_multilabel_original.txt', 'score_multinom_pos_org.txt','score_sgd_pos_org.txt','score_linear_svc_pos_org.txt','score_svc_pos_org.txt','score_majority_pos_org.txt',indices=idxs,train_indices=train_idxs,test_indices=test_idxs,dataset_num="10", pron_tag_choice=True, pred_tag_choice=True,tfidf_choice=False)
    #
    gender_clf(filenames_boys, filenames_girls,  'svm_11_pos_pron_pred_all_multilabel_original.txt', 'score_multinom_pos_org.txt','score_sgd_pos_org.txt','score_linear_svc_pos_org.txt','score_svc_pos_org.txt','score_majority_pos_org.txt',indices=idxs,train_indices=train_idxs,test_indices=test_idxs,dataset_num="11", pron_choice=True, pron_tag_choice=True, pred_choice=True, pred_tag_choice=True,tfidf_choice=False)
    gender_clf(filenames_boys, filenames_girls, 'svm_12_pos_pron_tag_nums_multilabel_original.txt', 'score_multinom_pos_org.txt','score_sgd_pos_org.txt','score_linear_svc_pos_org.txt','score_svc_pos_org.txt','score_majority_pos_org.txt',indices=idxs,train_indices=train_idxs,test_indices=test_idxs, dataset_num="12", pron_tag_choice=True, pron_choice=True, lix_choice=True, heylighen_fmeasure_choice=True,tfidf_choice=False)

    gender_clf(filenames_boys, filenames_girls,  'svm_13_pos_pred_tag_nums_multilabel_original.txt', 'score_multinom_pos_org.txt','score_sgd_pos_org.txt','score_linear_svc_pos_org.txt','score_svc_pos_org.txt','score_majority_pos_org.txt',indices=idxs,train_indices=train_idxs,test_indices=test_idxs,dataset_num="13", pred_tag_choice=True,pred_choice=True,lix_choice=True,heylighen_fmeasure_choice=True,tfidf_choice=False)

    gender_clf(filenames_boys, filenames_girls, 'svm_14_pos_pron_pred_tag_nums_multilabel_original.txt', 'score_multinom_pos_org.txt','score_sgd_pos_org.txt','score_linear_svc_pos_org.txt','score_svc_pos_org.txt','score_majority_pos_org.txt',indices=idxs,train_indices=train_idxs,test_indices=test_idxs, dataset_num="14", pron_tag_choice=True, pron_choice=True,pred_tag_choice=True, pred_choice=True, lix_choice=True, heylighen_fmeasure_choice=True,tfidf_choice=False)

    gender_clf(filenames_boys, filenames_girls, 'svm_2_pos_multilabel.txt','score_multinom_pos.txt','score_sgd_pos.txt','score_linear_svc_pos.txt','score_svc_pos.txt', 'score_majority_pos.txt',indices=idxs,train_indices=train_idxs,test_indices=test_idxs,  dataset_num="2",
               tfidf_choice=True)
    gender_clf(filenames_boys, filenames_girls, 'svm_3_pos_pron_multilabel.txt','score_multinom_pos.txt','score_sgd_pos.txt','score_linear_svc_pos.txt','score_svc_pos.txt', 'score_majority_pos.txt',indices=idxs,train_indices=train_idxs,test_indices=test_idxs,
               dataset_num="3", pron_choice=True, tfidf_choice=True)
    gender_clf(filenames_boys, filenames_girls, 'svm_4_pos_lix_multilabel.txt', 'score_multinom_pos.txt','score_sgd_pos.txt','score_linear_svc_pos.txt','score_svc_pos.txt', 'score_majority_pos.txt',indices=idxs,train_indices=train_idxs,test_indices=test_idxs,
               dataset_num="4", lix_choice=True, tfidf_choice=True)
    gender_clf(filenames_boys, filenames_girls, 'svm_5_pos_heylighen_fmeasure_multilabel.txt', 'score_multinom_pos.txt','score_sgd_pos.txt','score_linear_svc_pos.txt','score_svc_pos.txt','score_majority_pos.txt', indices=idxs,train_indices=train_idxs,test_indices=test_idxs,
                dataset_num="5", heylighen_fmeasure_choice=True, tfidf_choice=True)

    gender_clf(filenames_boys, filenames_girls, 'svm_6_pos_pred_multilabel.txt','score_multinom_pos.txt','score_sgd_pos.txt','score_linear_svc_pos.txt','score_svc_pos.txt', 'score_majority_pos.txt', indices=idxs,train_indices=train_idxs,test_indices=test_idxs,
               dataset_num="6", pred_choice=True, tfidf_choice=True)
    gender_clf(filenames_boys, filenames_girls, 'svm_7_pos_pron_tag_multilabel.txt','score_multinom_pos.txt','score_sgd_pos.txt','score_linear_svc_pos.txt','score_svc_pos.txt', 'score_majority_pos.txt', indices=idxs,train_indices=train_idxs,test_indices=test_idxs,
               dataset_num="7", pron_tag_choice=True, tfidf_choice=True)
    gender_clf(filenames_boys, filenames_girls, 'svm_8_pos_pred_tag_multilabel.txt', 'score_multinom_pos.txt','score_sgd_pos.txt','score_linear_svc_pos.txt','score_svc_pos.txt','score_majority_pos.txt', indices=idxs,train_indices=train_idxs,test_indices=test_idxs,
               dataset_num="8", pred_tag_choice=True, tfidf_choice=True)

    gender_clf(filenames_boys, filenames_girls, 'svm_9_pos_pron_pred_multilabel.txt','score_multinom_pos.txt','score_sgd_pos.txt','score_linear_svc_pos.txt','score_svc_pos.txt', 'score_majority_pos.txt', indices=idxs,train_indices=train_idxs,test_indices=test_idxs,
               dataset_num="9", pron_choice=True, pred_choice=True, tfidf_choice=True)
    gender_clf(filenames_boys, filenames_girls, 'svm_10_pos_pron_tag_pred_tag_multilabel.txt', 'score_multinom_pos.txt','score_sgd_pos.txt','score_linear_svc_pos.txt','score_svc_pos.txt', 'score_majority_pos.txt',indices=idxs,train_indices=train_idxs,test_indices=test_idxs,
                dataset_num="10", pron_tag_choice=True, pred_tag_choice=True, tfidf_choice=True)

    gender_clf(filenames_boys, filenames_girls, 'svm_11_pos_pron_pred_all_multilabel.txt','score_multinom_pos.txt','score_sgd_pos.txt','score_linear_svc_pos.txt','score_svc_pos.txt', 'score_majority_pos.txt', indices=idxs,train_indices=train_idxs,test_indices=test_idxs,
                dataset_num="11", pron_choice=True, pron_tag_choice=True, pred_choice=True,
               pred_tag_choice=True, tfidf_choice=True)
    gender_clf(filenames_boys, filenames_girls, 'svm_12_pos_pron_tag_nums_multilabel.txt','score_multinom_pos.txt','score_sgd_pos.txt','score_linear_svc_pos.txt','score_svc_pos.txt', 'score_majority_pos.txt', indices=idxs,train_indices=train_idxs,test_indices=test_idxs,
                dataset_num="12", pron_tag_choice=True, pron_choice=True, lix_choice=True,
               heylighen_fmeasure_choice=True, tfidf_choice=True)

    gender_clf(filenames_boys, filenames_girls, 'svm_13_pos_pred_tag_nums_multilabel.txt','score_multinom_pos.txt','score_sgd_pos.txt','score_linear_svc_pos.txt','score_svc_pos.txt', 'score_majority_pos.txt',indices=idxs,train_indices=train_idxs,test_indices=test_idxs,
                dataset_num="13", pred_tag_choice=True, pred_choice=True, lix_choice=True,
               heylighen_fmeasure_choice=True, tfidf_choice=True)

    gender_clf(filenames_boys, filenames_girls, 'svm_14_pos_pron_pred_tag_nums_multilabel.txt','score_multinom_pos.txt','score_sgd_pos.txt','score_linear_svc_pos.txt','score_svc_pos.txt', 'score_majority_pos.txt', indices=idxs,train_indices=train_idxs,test_indices=test_idxs,
                dataset_num="14", pron_tag_choice=True, pron_choice=True, pred_tag_choice=True,
               pred_choice=True, lix_choice=True, heylighen_fmeasure_choice=True, tfidf_choice=True)