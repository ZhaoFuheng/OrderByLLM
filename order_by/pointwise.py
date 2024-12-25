class Pointwise_Key:
    def __init__(self, key):
        self.key = key
    def value(self, client, prompt, modelname, output_type):
        api_calls = 0
        while api_calls < 3:
            api_calls += 1
            try:
                response = client.chat.completions.create(
                    model = modelname,
                    messages=[{"role": "user", "content": prompt}],
                    n = 1,
                    temperature=0.0,
                    max_tokens=4096,
                )
                responses = [choice.message.content.strip() for choice in response.choices]
                return output_type(responses[0]), api_calls
            except Exception as e:
                print(e)
        return None, api_calls 
