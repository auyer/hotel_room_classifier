# Places 2 Easy Mode

This script will turn any Places dataset to the "easy mode" format.

It reades the categories and validation files (`categories_places365.txt`, `places365_val.txt`), and the train file, that will change for each dataset version (`places365_train_challenge.txt`).

There are a few parameters that might need to change for each run: 

```
NumClass = 365
Workers = 18   # this is more affected by the ammount of files that can be open at once thant the ammount of thread available
CategoriesFile = "categories_places365.txt"
TrainFile = "places365_train_challenge.txt"
ValFile = "places365_val.txt"
```