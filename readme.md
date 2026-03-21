



```sh
uvicorn app.main:app --reload
```


```sh
curl -X POST "http://127.0.0.1:8000/run" -F "file=@multi_agent\data\Assembly - 1.xlsx"
  ```