# Home Credit Default Risk Competition - 27th Place Solution

- [Competition Site](https://www.kaggle.com/c/home-credit-default-risk)
- [Presentation (Japanese)](https://docs.google.com/presentation/d/1JARQhDDki6rf8za6loAbDFNRI87sOfM4o4oqeHcDIbg/edit?usp=sharing)

## Requirements
- LightGBM
- XGBoost
- feather-format
- Keras

## Run Single Model

```bash
cd feature
python feature.py nocache
cd ../model
python lgbm.py
```
