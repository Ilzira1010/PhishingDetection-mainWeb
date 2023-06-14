import re
import os
import nltk
import string
import mailbox
import pandas as pd
from bs4 import BeautifulSoup
from collections import Counter
from nltk.corpus import stopwords
from email.header import decode_header
from nltk.tokenize import word_tokenize

import warnings
from bs4 import GuessedAtParserWarning

warnings.filterwarnings('ignore', category=GuessedAtParserWarning)

phishing_emails = mailbox.mbox(r'C:\Users\ilzira\Desktop\PythonProjects\PhishingDetection-mainWeb\phishing-mails.mbox')
valid_emails = mailbox.mbox(r"C:\Users\ilzira\Desktop\PythonProjects\PhishingDetection-mainWeb\legal-mails.mbox")


class EmailParser:
    urlRegex = r'https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=;]*)'
    emailRegex = r'([a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)'

    def __init__(self, email):
        self.email = email
        self.__extract_email_parts()

    def __extract_email_parts(self):
        no_of_attachments = 0
        text = str(self.email['Subject']) + " "
        htmlDoc = ""
        for part in self.email.walk():
            content_type = part.get_content_type()
            if content_type == 'text/plain':
                text += str(part.get_payload())
            elif content_type == 'text/html':
                htmlDoc += part.get_payload()
            else:
                main_content_type = part.get_content_maintype()
                if main_content_type in ['image', 'application']:
                    no_of_attachments += 1
        self.text, self.html, self.no_of_attachments = text, htmlDoc, no_of_attachments

    def get_urls(self):
        text_urls = set(re.findall(EmailParser.urlRegex, self.text))
        html_urls = set(re.findall(EmailParser.urlRegex, self.html))
        return list(text_urls.union(html_urls))

    def get_email_text(self):
        if (self.html != ""):
            soup = BeautifulSoup(self.html)
            self.text += soup.text
        return self.text

    def get_no_of_attachments(self):
        return self.no_of_attachments

    def get_sender_email_address(self):
        sender = email['From']
        try:
            emails = re.findall(EmailParser.emailRegex, sender)
        except:
            h = decode_header(email['From'])
            header_bytes = h[0][0]
            sender = header_bytes.decode('ISO-8859-1')
            emails = re.findall(EmailParser.emailRegex, sender)
        if (len(emails) != 0):
            return emails[len(emails) - 1]
        else:
            return ''


class StringUtil:
    dotRegex = r'\.'
    digitsRegex = r'[0-9]'
    ipAddressRegex = r'(?:[0-9]{1,3}\.){3}[0-9]{1,3}'
    dashesRegex = r'-'
    specialCharsRegex = r'[()@:%_\+~#?\=;]'
    words = Counter()
    stop_words = set(stopwords.words('english'))
    stemmer = nltk.PorterStemmer()
    punctuations = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '_', '=', '+', ';', ':', "'", '"', '?', '/',
                    '<', '>', '.', ',', '/', '~', '`']

    # This method takes a list of URLs as input and processes them.
    # It counts the number of dots, dashes, and special characters in each URL
    def process_urls(self, urls):
        noOfDots, noOfDashes, noOfSpecialChars, hasIpAddressInUrl, noOfIpAddress, noOfHttpsLinks = 0, 0, 0, 0, 0, 0
        for url in urls:
            if url.startswith('https://'):
                noOfHttpsLinks += 1
            noOfDots += len(re.findall(StringUtil.dotRegex, url))
            noOfDashes += len(re.findall(StringUtil.dashesRegex, url))
            noOfSpecialChars += len(re.findall(StringUtil.specialCharsRegex, url))
            noOfIpAddress += len(re.findall(StringUtil.ipAddressRegex, url))
        if noOfIpAddress > 0:
            hasIpAddressInUrl = 1
        return len(urls), noOfDots, noOfDashes, noOfSpecialChars, hasIpAddressInUrl, noOfIpAddress, noOfHttpsLinks

    # This method takes a string of text as input and processes it. It performs several operations on the text, including converting it to lowercase,
    # removing escape sequences, removing punctuation and digits, tokenizing the text into individual words
    def process_text(self, text):
        text = text.lower()  # lowercase
        text = re.sub(r'[\n\t\r]', ' ', text)  # remove escape sequences

        # remove punctuations
        punctuation = string.punctuation  # Get all punctuation marks
        translator = str.maketrans('', '',
                                   punctuation + string.digits)  # Create a translator to remove punctuation and digits
        text = text.translate(translator)  # Remove punctuation and digits using translate()

        # tokenize and stem words
        word_tokens = word_tokenize(text)
        filtered_text = []
        for w in word_tokens:
            if w not in StringUtil.stop_words:
                filtered_text.append(w)

        # count frequency of words
        word_counts = Counter(filtered_text)
        stemmed_word_count = Counter()
        for word, count in word_counts.items():
            stemmed_word = StringUtil.stemmer.stem(word)
            stemmed_word_count[stemmed_word] += count
        word_counts = stemmed_word_count
        StringUtil.words += word_counts
        return word_counts

    # This method takes an email address as input and processes it. It calculates various metrics related to the email address,
    # including its length, the counts of dots, dashes, special characters, digits, and subdomains
    def process_email_address(self, emailid):
        length, noOfDots, noOfDashes, noOfSpecialChars, noOfDigits, noOfSubdomains = 0, 0, 0, 0, 0, 0

        length = len(emailid)
        if (length > 0):
            username, domain = emailid.split('@')
            noOfSubdomains = len(re.findall(StringUtil.dotRegex, domain)) - 1
            noOfDots = len(re.findall(StringUtil.dotRegex, username))
            noOfSpecialChars = len(re.findall(StringUtil.specialCharsRegex, username))
            noOfDashes = len(re.findall(StringUtil.dashesRegex, emailid))
            noOfDigits = len(re.findall(StringUtil.digitsRegex, emailid))

        return length, noOfDots, noOfDashes, noOfSpecialChars, noOfDigits, noOfSubdomains

        # This method returns the 1000 most common words encountered so far in the text processing.
        # It accesses the class variable StringUtil.words,
        # which is a Counter object that keeps track of word frequencies across all processed texts.

    def get_most_common_words(self):
        return StringUtil.words.most_common(1000)


# StringUtil class and the EmailParser class to process a list of phishing emails and extract relevant features.
# It then adds the extracted features along with the corresponding class label to a pandas DataFrame named df1

df1 = pd.DataFrame(
    columns=['text', 'lengthOfEmailId', 'noOfDotsInEmailId', 'noOfDashesInEmailId', 'noOfSpecialCharsInEmailId',
             'noOfDigitsInEmailId', 'noOfSubdomainsInEmailId', 'noOfUrls', 'noOfDotsInUrls', 'noOfDashesInUrls',
             'noOfSpecialCharsInUrls', 'hasIpAddressInUrls', 'noOfIpAddressInUrls', 'noOfHttpsLinks',
             'no_of_attachments', 'class_label'])
stringUtil = StringUtil()
for email in phishing_emails:
    emailParser = EmailParser(email)
    no_of_attachments = emailParser.get_no_of_attachments()
    emailid_features = stringUtil.process_email_address(emailParser.get_sender_email_address())
    urls_features = stringUtil.process_urls(emailParser.get_urls())
    word_dict = stringUtil.process_text(emailParser.get_email_text())
    df1.loc[len(df1)] = [word_dict, emailid_features[0], emailid_features[1], emailid_features[2], emailid_features[3],
                         emailid_features[4], emailid_features[5], urls_features[0], urls_features[1], urls_features[2],
                         urls_features[3], urls_features[4], urls_features[5], urls_features[6], no_of_attachments, 1]

# get most common words from phishing emails
malicious_words = stringUtil.get_most_common_words()

# shows the usage of the StringUtil class and the EmailParser class to process a list of valid emails and extract relevant features.
# It then adds the extracted features along with the corresponding class label to a new pandas DataFrame named

df2 = pd.DataFrame(
    columns=['text', 'lengthOfEmailId', 'noOfDotsInEmailId', 'noOfDashesInEmailId', 'noOfSpecialCharsInEmailId',
             'noOfDigitsInEmailId', 'noOfSubdomainsInEmailId', 'noOfUrls', 'noOfDotsInUrls', 'noOfDashesInUrls',
             'noOfSpecialCharsInUrls', 'hasIpAddressInUrls', 'noOfIpAddressInUrls', 'noOfHttpsLinks',
             'no_of_attachments', 'class_label'])
stringUtil = StringUtil()
for email in valid_emails:
    emailParser = EmailParser(email)
    no_of_attachments = emailParser.get_no_of_attachments()
    emailid_features = stringUtil.process_email_address(emailParser.get_sender_email_address())
    urls_features = stringUtil.process_urls(emailParser.get_urls())
    word_dict = stringUtil.process_text(emailParser.get_email_text())
    df2.loc[len(df2)] = [word_dict, emailid_features[0], emailid_features[1], emailid_features[2], emailid_features[3],
                         emailid_features[4], emailid_features[5], urls_features[0], urls_features[1], urls_features[2],
                         urls_features[3], urls_features[4], urls_features[5], urls_features[6], no_of_attachments, 0]

df = pd.concat([df1, df2], axis=0)

# adds a new column to the DataFrame df named 'noOfMaliciousWords'
# and populates it with the count of malicious words present in each email.
# It then removes the 'text' column from the DataFrame.

df['noOfMaliciousWords'] = df['text'].apply(
    lambda x: len(set(x.keys()).intersection(set(dict(malicious_words).keys()))))
df = df.drop(columns=['text'])

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score

x = df.drop(columns=["class_label"]).values
y = df["class_label"].values

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=34, test_size=0.2)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

# This line calculates and prints the prediction accuracy of the model by comparing the predicted labels y_pred
# with the true labels y_test using the accuracy_score() function from sklearn.metrics.
# The accuracy score represents the proportion of correctly predicted labels.
print('Prediction accuracy: ', accuracy_score(y_test, y_pred))

# This line calculates and prints the precision score of the model by comparing the predicted labels y_pred
# with the true labels y_test using the precision_score() function from sklearn.metrics.
# The precision score represents the ability of the classifier to correctly identify positive instances.
print('Precision Score: ', precision_score(y_test, y_pred))
