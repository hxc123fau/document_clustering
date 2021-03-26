import json
import nltk

# with open('NLP-test.json','rb') as f:
#     data = json.loads(f.read())

# print('data',type(data))


def get_json_value_by_key(in_json, target_key, results=[]):
    if isinstance(in_json, dict):  # if input is dict
        for key in in_json.keys():  # loop get key
            data = in_json[key]
            get_json_value_by_key(data, target_key, results=results)  # embedding

            if key == target_key:
                if data==None:
                    # print('aaa')
                    data = in_json["abstract"]
                    results.append(data)
                else:
                    # print('bbb')
                    results.append(data)

    elif isinstance(in_json, list) or isinstance(in_json, tuple):  # if input is list or tuple
        for data in in_json:  # 循环当前列表或元组
            get_json_value_by_key(data, target_key, results=results)  # embedding

    return results


# res = get_json_value_by_key(data, "fulltext")
# res = get_json_value_by_key(data, "abstract")
# print('res',len(res))
#
# for j,j_text in enumerate(res):
#     d1 = nltk.sent_tokenize(j_text)
#     text = []
#     for i in d1:
#         w1 = nltk.word_tokenize(i)
#         text.extend(w1)

    # print('text',j, len(text),text)
