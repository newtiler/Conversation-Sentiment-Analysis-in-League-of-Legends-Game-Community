from flask import Flask, render_template, request
import pandas as pd
from pythainlp import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from collections import Counter
from pyvis.network import Network
import pythainlp
from pythainlp.corpus.common import thai_words
from pythainlp import Tokenizer
import re
from flask import Flask, render_template, request, redirect, url_for
app = Flask(__name__)

# Load the initial DataFrame
df = pd.read_excel('comment sentiment (masking).xlsx')
data_toxic_words = pd.read_csv('neg_words.txt', header=None, names=['text'])
data_positive_word = pd.read_csv('pos_words.txt', header=None, names=['text'])
toxic_words = []
positive_word = []
toxic_words.extend(data_toxic_words.values.flatten())
positive_word.extend(data_positive_word.values.flatten())

words = set(thai_words())  # thai_words() returns frozenset
words.update(toxic_words)
custom_tokenizer = Tokenizer(words)
toxic_words = list(set(toxic_words))
positive_word = list(set(positive_word))
def custom_token(text):
    text = re.sub(r'[^\u0E00-\u0E7Fa-zA-Z0-9\s]', '', text) # remove special characters and emojis
    text = re.sub(r'\s+', ' ', text) # remove extra whitespace
    text = re.sub(r':[a-z_]+:', '', text) # remove all emoji
    text = text.strip() # remove leading and trailing whitespace
    tokens = custom_tokenizer.word_tokenize(text)
    return tokens
df['tokens'] = df['content'].apply(custom_token)
df['tokens'] = df['tokens'].apply(lambda x: ' '.join(x))
dict_score = {"อีควาย":-3, "จิ๋ม":-3, "ไอ้ส้นตีน":-3, "ปลวก":-1, "กระจุกกระจิก":-1,
"แย่มาก":-3, "กดขี่":-2, "ขึ้แตก":-2, "ขี้ขลาดตาขาว":-2, "ฟักยู":-3, "ทราม":-2,
"บ่อนทำลาย":-3, "แย่จัง":-3, "อีปลาเงือก":-3, "ข่มขวัญ":-1, "ขายตัว":-3, "ไอ้ขึ้หมา":-2,
"กวนตีน":-3, "ทมิฬ":-3, "คอร์รัปชัน":-2, "เสแสร้ง":-1, "ไอ้บ้า":-1, "ขายขี้หน้า":-2,
"กระดอ":-3, "อีอับปรี":-2, "ทุจริต":-2, "อีเหี้ย":-3, "แซะ":-1, "ฆ่าแกง":-3, "โกหก":-1,
"เบี๊อก":-2, "เอ๋อ":-2, "มึง":-1, "เสือก":-3, "เจี๊ยว":-1, "ฟัค":-3, "เกเร":-2, "ต่ำตม":-3,
"ย่ำแย่":-3, "แย่":-2, "เหี้ย":-3, "พ่อง":-3, "อีกะหรี":-3, "ซ้ำร้าย":-2, "แมงดา":-3, "ดูถูก":-1,
"หน้าโง่":-3, "ชั่งแม่ง":-3, "ตอแหล":-3, "แม่ง":-3, "ห่วย":-3, "ฟาย":-3, "เยสเเม่":-3,
"เสือกพะยูน":-3, "เหียก":-1, "อีหมา":-2, "ขัดใจ":-1, "ดูหมิ่น":-2, "พ่อมึง":-3, "อีห่า":-3,
"แทงกัน":-3, "ร้าย":-1, "ทุบตี":-2, "ไอสัด":-3, "โสโครก":-3, "ขัดขืน":-2, "ตัวแสบ":-1,
"กลั่นแกล้ง":-2, "วะ":-1, "คดโกง":-2, "อุบาทว์":-3, "โหดร้าย":-2, "กล่าวหา":-1, "เฉโก":-1,
"สัส":-3, "ไอ้เวร":-3, "ไอ้หมา":-3, "หรรม":-2, "เบื่อหน้า":-2, "บ้า":-1, "ฆ่า":-3,
"สันดาน":-3, "ลูกโสเภณี":-3, "ไม่เจียมกะลาหัว":-3, "แตด":-2, "ทะลึ่ง":-2, "เพี้ยน":-1,
"ขโมย":-2, "ประจาร":-2, "กวาดล้าง":-2, "ความขุ่นเคือง":-2, "เงี่ยน":-3, "เบื๊อก":-2,
"หน้าส้นตีน":-3, "แค้น":-2, "หน้าตัวเมืย":-3, "เสื่อมเสีย":-2, "ตราหน้า":-2, "เสียใจ":-1,
"กดราคา":-1, "ต่ำต้อย":-2, "ข่ม":-1, "หน้าควย":-3, "ตำหนิ":-2, "ปี้กัน":-3, "วิกฤติ":-2,
"ฉิบหาย":-3, "ปาราชิก":-1, "ล้าสมัย":-2, "มารศาสนา":-3, "รุกรุย":-1, "กระทำชำเรา":-3,
"อีสัตว์":-3, "ไอ้สัตว์":-3, "แม่มึงตาย":-3, "รุม":-1, "ชิงหมาเกิด":-2, "เกรียน":-1,
"เสื่อมโทรม":-3, "หำ":-2, "ไอเหี้ยหน้าหี":-3, "ลูกอีดอกทอง":-3, "ถุย":-1, "อีขี้ข้า":-3,
"มั่วนิ่ม":-2, "อีฟันหมาบ้า":-3, "ไอเหี้ย":-3, "เฆี่ยน":-2, "ไอ้ลาว":-3, "อีชาติชั่ว":-3,
"สิ้นสติ":-2, "กดดัน":-1, "แรด":-3, "เซ่อ":-2, "ปรัมปรา":-1, "ประชด":-2, "อีอ้วน":-3,
"xี":-3, "ไร้สาระ":-2, "ซวย":-2, "ถึงฆาต":-3, "อีดอกทอง":-3, "หน้าหี":-3, "ไม่ดี":-2,
"เกลียด":-2, "สับปลับ":-1, "ง่าว":-2, "ลำบาก":-2, "ไม่เอาไหน":-2, "ขืนใจ":-3,
"ขวางโลก":-3, "จู๋":-1, "กาก":-3, "อีร้อย":-1, "มั่ว":-2, "ฆ่าฟัน":-3, "แดก":-3,
"อีหน้าควาย":-3, "ประจาน":-2, "อื้อฉาว":-2, "โสมม":-2, "ปากเสีย":-2, "ชิงชัง":-2,
"ยิงกัน":-3, "อยากเอาหญิง":-3, "ดัดสันดาน":-3, "ไอเข้":-2, "ล่อกัน":-2, "อีช้างเย๊ด":-3,
"บกพร่อง":-2, "นรกแดกกบาล":-3, "ปากหมา":-3, "เนรคุณ":-3, "ทุเรศ":-3, "อีดำ":-3,
"ฆ่าล้างโคตร":-3, "ทรยศ":-2, "ความฉิบหาย":-3, "ห่า":-3, "อีดอก":-3, "ขัดแย้ง":-2,
"ตีกัน":-1, "จังไร":-3, "สะเหร่อ":-3, "อกตัญญู":-2, "กะหรี่":-3, "ขายชาติ":-3, "อีสัด":-3,
"ก่อกวน":-1, "ไอ้":-1, "บาดหมาง":-2, "กระจอก":-3, "อีหน้าหมา":-3, "ตัดพ้อ":-2,
"ด่า":-3, "โกง":-3, "พังพินาศ":-3, "หว่ะ":-1, "เปรต":-3, "บิดเบือน":-2, "ความชั่วช้า":-3,
"ร้ายกาจ":-2, "สาด":-3, "ความขมขื่น":-3, "ไอ้เวร":-3, "อีเวร":-3, "ไอเวร":-3,
"ปัญญาอ่อน":-3, "สาส":-3, "สัด":-3, "รังเกียจ":-3, "บรรลัย":-2, "โหดเหี้ยม":-3,
"อีชาติชั่ว":-3, "แม่มึง":-3, "เสร่อ":-3, "เยี่ยว":-3, "หมาบ้า":-2, "เซ็ง":-1, "ช้างเย็ด":-3, "เหี้ยมโหด":-3,
"ตาย":-3, "ควย":-3, "ชาติหมา":-3, "ขู่":-2, "เชี่ย":-3, "เดือดร้อน":-2, "ขายหน้า":-2,
"ประสาท":-3, "ไอ้สัตว์นรก":-3, "ตกม้าตาย":-3, "เห็บ":-2, "เศษนรก":-3, "เบียดเบียน":-2,
"จัญไร":-3, "เสนียด":-3, "เผด็จการ":-2, "ผิดหวัง":-2, "อีหน้าหี":-3, "ส้นตีน":-3,
"ประชดประชัน":-2, "เฮงซวย":-3, "ขี้":-3, "ขูดรีด":-2, "ข่มขืน":-3, "หน้าด้าน":-3,
"น่าเบื่อ":-2, "จรวย":-3, "ตบตา":-1, "อัปปรี":-3, "เบื่อ":-2, "แสรด":-3, "ชั่ว":-3,
"ตกนรกทั้งเป็น":-3, "ใจหมา":-3, "ด้อย":-2, "หี":-3, "สันหลังยาว":-2, "แช่ง":-3,
"ลูกอีสาด":-3, "ละเหี่ยใจ":-2, "มั่วซั่ว":-1, "โง่":-3, "นอกใจ":-2, "ข่มขู่":-2, "ดัดจริต":-2,
"พ่อมึงตาย":-3, "กวนส้นตีน":-3, "กระทืบ":-3, "ด่าทอ":-2, "กรี่":-3, "ไอ้สัส":-3,
"ฉุนเฉียว":-3, "สวะ":-3, "ผู้หญิงต่ำๆ":-3, "ไอ้ควาย":-3, "ร่าน":-3, "ฆ่ายกครัว":-3,
"ระยำ":-3, "เลว":-3, "เฉื่อย":-1, "กระทำอนาจาร":-3, "หมองมัว":-1, "ชาติชั่ว":-3,
"ลูกอีกะหรี่":-3, "สถุล":-3, "ควาย":-3, "ไม่ได้เรื่อง":-2, "ยัดแม่":-3, "เคราะห์ซ้ำกรรมซัด":-2,
"อีกระหรี่":-3, "ขู่เข็ญ":-2, "ขยะ":-3, "หงุดหงิด":-2, "ตายห่า":-3, "ชิปหาย":-3, "ชิบหาย":-3,
"รำคาญ":-2, "ลำคาน":-2, "อุบาท":-3, "ไม่มีปัญญา":-3, "ตุ๊ด":-3, "กระเทย":-3, "ngo":-3,
"ประสาทกิน":-3, "แจก":-1, "หัวร้อน":-2, "รวดเร็ว":2, "ได้ดี":2, "ไม่ผิดหวัง":2, "โครตดี":3, "โครตหมาะ":3,
"ไม่เคยผิดหวัง":3, "สบาย":2, "ยอดเยี่ยม":3, "Love":3, "ปลื้มปริ่ม":3, "ถูกใจ":2,
"รอบคอบ":2, "เร็ว":1, "ละเอียด":1, "ละมุน":1, "สุภาพ":1, "อร่อย":2, "อ่อนโยน":1,
"อัศจรรย์":3, "ดีใจ":3, "อุ่นใจ":2, "รัก":3, "ชอบ":3, "งดงาม":3, "แจ่ม":3, "แจ่มใส":3,
"ชัดแจ๋ว":1, "น่ารื่นรมย์":3, "น่ารัก":2, "ฉลาด":2, "อบอุ่น":1, "ความสุข":3, "สุข":3,
"ประทับใจ":3, "ซาบซึ้ง":2, "ภูมิใจ":2, "มหัศจรรย์ใจ":3, "มีความยินดี":3, "ผ่องใส":2,
"ยอดเยี่ยม":3, "เยี่ยม":3, "เยี่ยมยอด":3, "โดดเด่น":3, "รุ่ง":2, "รุ่งเรือง":2, "รุ่งโรจน์":2,
"ทันสมัย":2, "ล้ำเลิศ":2, "เสถียรภาพ":1, "เพลิดเพลิน":2, "น่านับถือ":2, "สดใส":2, "สบาย":2,
"สมจริง":1, "สมบูรณ์":2, "สร้างสรรค์":2, "สวย":2, "สวยงาม":3, "เหมาะเจาะ":2, "มีประโยชน์":2,
"สุขสันต์":3, "สนุก":3, "พอใจ":1, "น่าพอใจ":2, "น่าประทับใจ":3, "น่าพึงพอใจ":2, "น่าฟัง":2,
"น่าภาคภูมิใจ":3, "น่ามหัศจรรย์":3, "น่ายกย่อง":3, "น่ายินดี":3, "สนุกสนาน":3, "รวดเร็วทันใจ":2,
"ดีมาก":3, "สุดยอด":3
}
dict_score = dict(dict_score)
# Set the score of each word definition.
def def_score(df, dict_score):
    score = 0
    words = df.lower().split(" ")
    for word in words:
        if word in dict_score:
            score += int(dict_score[word])

    return score
scores = []
for d in df['tokens']:
    score = def_score(d, dict_score)
    scores.append(score)
df['definition_feeling_score'] = scores

df['weight_label'] = ''
df.loc[df['definition_feeling_score'] > 0, 'weight_label'] = 'pos'
df.loc[df['definition_feeling_score'] < 0, 'weight_label'] = 'neg'
df.loc[df['definition_feeling_score'] == 0, 'weight_label'] = 'neu'
scores = []
for d in df['tokens']:
    score = def_score(d, dict_score)
    scores.append(score)
df['tokens'].fillna('', inplace=True)

# Create a TF-IDF vectorizer
vectorizer = CountVectorizer()

# Convert text data to feature vectors
X = vectorizer.fit_transform(df['tokens'])

X_train, X_test, y_train, y_test = train_test_split(X, df['weight_label'], test_size=0.3, random_state=42)
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)
df.drop(["label", "tokens", "definition_feeling_score","id"], axis=1)
from collections import Counter
import pandas as pd
from pyvis.network import Network

# Assuming you have a DataFrame named df with columns "author", "target", and "label"

# Separate data into lists
authors = list(df["author"])
targets = list(df["target"])
labels = list(df["weight_label"])

# Use Counter to count occurrences of author-target pairs
edge_counts = Counter(zip(authors, targets, labels))

# Create a pyvis network
net = Network(height="750px", width="100%", notebook=True)

# Add nodes to the network
for author in authors:
    net.add_node(author, label=author)

# Add edges to the network, skipping NaN target values
for (author, target, label), count in edge_counts.items():
    if pd.notna(target) and target != 'NaN':  # Check for NaN and 'NaN'
        # Define a color based on the label
        color = None
        if label == 'neg':
            color = 'red'
        elif label == 'pos':
            color = 'green'
        elif label == 'neu':
            color = 'grey'

        # Use count as the value for the edge width
        # Add the arrows attribute to create arrowheads
        net.add_edge(author, target, value=count, color=color, arrows="to", width=3, head_width=2, length_includes_head=True)

        net.add_edge(target, author, value=count, color=color, arrows="to", width=3, head_width=2, length_includes_head=True)

# Show the network
net.show("network_loldrama.html")
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/add_data', methods=['POST'])
def add_data():
    author = request.form['author']
    content = request.form['content']
    target = request.form.get('target', '')  # Use request.form.get to handle optional target

    tokens = custom_token(content)
    tokens = ' '.join(tokens)
    score = def_score(tokens, dict_score)
    weight_label = 'pos' if score > 0 else 'neg' if score < 0 else 'neu'

    new_data = {'author': [author], 'content': [content], 'target': [target], 'tokens': [tokens],
                'definition_feeling_score': [score], 'weight_label': [weight_label]}
    new_df = pd.DataFrame(new_data)

    # Append new data to the existing DataFrame
    global df
    df = pd.concat([df, new_df], ignore_index=True)

    new_data_to_display = new_df.drop(["tokens", "definition_feeling_score"], axis=1)


    # Update the network visualization
    update_network(df)

    return render_template('index.html', data=new_data_to_display.to_html(classes='table table-bordered table-striped'))

def update_network():
    authors = list(df["author"])
    targets = list(df["target"])
    labels = list(df["weight_label"])

    edge_counts = Counter(zip(authors, targets, labels))

    net = Network(height="750px", width="100%", notebook=True)

    # Add authors and targets as nodes
    for author in authors:
        net.add_node(author, label=author)
    for target in targets:
        if pd.notna(target):
            net.add_node(target, label=target)

    # Add edges connecting authors to targets
    for (author, target, label), count in edge_counts.items():
        if pd.notna(target):
            color = None
            if label == 'neg':
                color = 'red'
            elif label == 'pos':
                color = 'green'
            elif label == 'neu':
                color = 'grey'

            net.add_edge(author, target, value=count, color=color, arrows="to", width=3, head_width=2,
                         length_includes_head=True)

    net.save_graph("static/network_loldrama.html")

@app.route('/filter_data', methods=['POST'])
def filter_data():
    filter_value = request.form.get('filter')

    if filter_value == 'neg':
        filtered_df = df[df['weight_label'] == 'neg']
    elif filter_value == 'pos':
        filtered_df = df[df['weight_label'] == 'pos']
    else:
        filtered_df = df

    # Update the network visualization with the filtered data
    update_network(filtered_df)

    return redirect(url_for('index'))  # Redirect back to the homepage without displaying the filtered table

# Modify your existing update_network function to accept a DataFrame
def update_network(filtered_df):
    authors = list(filtered_df["author"])
    targets = list(filtered_df["target"])
    labels = list(filtered_df["weight_label"])

    edge_counts = Counter(zip(authors, targets, labels))

    net = Network(height="750px", width="100%", notebook=True)

    # Add authors and targets as nodes
    for author in authors:
        net.add_node(author, label=author)
    for target in targets:
        if pd.notna(target):
            net.add_node(target, label=target)

    # Add edges connecting authors to targets
    for (author, target, label), count in edge_counts.items():
        if pd.notna(target):
            color = None
            if label == 'neg':
                color = 'red'
            elif label == 'pos':
                color = 'green'
            elif label == 'neu':
                color = 'grey'

            net.add_edge(author, target, value=count, color=color, arrows="to", width=3, head_width=2,
                         length_includes_head=True)

    net.save_graph("static/network_loldrama.html")
if __name__ == '__main__':
    app.run(debug=True)
