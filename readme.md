



```sh
uvicorn app.main:app --reload
```


```sh
curl -X POST "http://127.0.0.1:8000/run" -F "file=@multi_agent\data\Assembly - 5.xlsx"
---
curl -X POST "http://127.0.0.1:8000/run" -F "folder_path=@multi_agent\data"

  ```


# Let's start with a picture of the neural network itself.
  ![1774095371556](image/README/1774095371556.png)


----
# Now a diagram of the complete pipeline — what happens from the moment the file comes in until a response is received.
![1774095646213](image/README/1774095646213.png)


-----~~~
```sh

```