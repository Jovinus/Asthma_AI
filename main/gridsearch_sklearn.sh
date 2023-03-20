python ./model/gridsearch_sklearn.py --feature fev --model logistic --n_jobs -1
python ./model/gridsearch_sklearn.py --feature fev --model rf --n_jobs -1
python ./model/gridsearch_sklearn.py --feature fev --model svm --n_jobs -1
python ./model/gridsearch_sklearn.py --feature fev --model ann --n_jobs -1

python ./model/gridsearch_sklearn.py --feature fev_fvc --model logistic --n_jobs -1
python ./model/gridsearch_sklearn.py --feature fev_fvc --model rf --n_jobs -1
python ./model/gridsearch_sklearn.py --feature fev_fvc --model svm --n_jobs -1
python ./model/gridsearch_sklearn.py --feature fev_fvc --model ann --n_jobs -1

python ./model/gridsearch_sklearn.py --feature imputed --model logistic --n_jobs -1
python ./model/gridsearch_sklearn.py --feature imputed --model rf --n_jobs -1
python ./model/gridsearch_sklearn.py --feature imputed --model svm --n_jobs -1
python ./model/gridsearch_sklearn.py --feature imputed --model ann --n_jobs -1