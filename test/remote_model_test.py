from langserve import RemoteRunnable

# ngrok remote 주소 설정
ngrok_url = 'https://b1ef-220-75-173-230.ngrok-free.app/prompt'

chain = RemoteRunnable("https://b1ef-220-75-173-230.ngrok-free.app/prompt")

for token in chain.stream({"topic": "딥러닝에 대해서 알려줘"}):
    print(token, end="")