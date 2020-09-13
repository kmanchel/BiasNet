import requests
from pprint import pprint

url = "http://localhost:5000/predict_API"
TEXT = " One man called in to comment, starting off with a criticism of CNN, lambasting the outlet for a chyron from last week that read, Fiery but mostly peaceful protests after police shooting, while buildings burned in the background behind a correspondent reporting from Kenosha, Wisconsin. The caller went on to refer to Stelter as 'Humpty Dumpty' and 'a stooge,' adding, We all know you're not reliable. According to Fox News, Stelter admitted the chyron was a mistake. I don't know who wrote it, probably a young producer who's trying their best under deadline in a breaking news situation, he said. That kind of thing becomes easily criticized and probably not the right banner to put on the screen. Another man called in to confront Stelter, saying, You guys always talk about how many times Trump has lied. Ive calculated that, I think with your chyrons ...Yeah, I dont know if theres any journalists left at CNN, but I know that if I were to estimate, about 300 different distortions or misinformation that we get out of CNN — and you have to watch them in the airport, which is harsh — but if you added all that up to 46 months, it comes out to be 300,000-plus distortions of truth. So, my thing is, you guys — this is how low you will go, the caller continued. You went out, and you made lies, and you defamed a child, referring to Covington (Kentucky) Catholic High School student Nick Sandmann, with whom the network settled a multimillion-dollar lawsuit. The caller added, And I dont believe in dividing our nation, it hurts our great nation, and so CNN is really the enemy of the truth. Thats my opinion, thank you. "
request = {"model_id": "2", "text": TEXT}

print("REQUEST:")
pprint(request)

r = requests.post(url, json=request)

print("RESPONSE:")
pprint(r.json())
