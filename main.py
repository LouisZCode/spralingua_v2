from agents import test_agent

test_message = "Dime un chiste sobre san luis potosi"

result = test_agent.invoke(
    {"messages":[{"role": "user", "content": test_message}]}
)

for i, msg in enumerate(result["messages"]):
    msg.pretty_print()