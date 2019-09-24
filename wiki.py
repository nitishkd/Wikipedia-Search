from nltk.tokenize import word_tokenize
import string
import xml.sax
import re
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer 
import heapq

class Page:
    def __init__(self):
        self.title = ""
        self.text = ""

class Processor:

    def __init__(self):
        print("processor intiated.")
        self.document_id = 0
        self.file_number = 0
        self.threshold = 5000 # number of docs to process before writing into II
        self.merge_threshold = 1000 # size of splitted II
        self.merge_file_num = 0
        self.doc_id_dump = []
        self.doc_id_dump_count = 0
        self.doc_id_threshold = 1000
        self.merge_doc_num = 0
        self.invertedIndex = []
        for i in range(6):
            self.invertedIndex.append({})

        self.level2_index = []
        for i in range(6):
            self.level2_index.append([])
        self.new_file = True

        self.mapping = ['title', 'body' , 'category', 'infobox' , 'references', 'ext_links']

    def tokenize(self, data):                                              #Tokenise
        tokenisedWords=re.findall("\d+|[\w]+",data)
        return tokenisedWords

    def compress(self, data):
        result = {}
        for item in data:
            if item in result:
                result[item] += 1
            else:
                result[item] = 1
        return result

    def dump_doc_id_doc_name(self):
        fp = open(str(self.doc_id_dump_count) + "_doc_id_doc_name_mapping.txt", 'w+')
        for (key, value) in self.doc_id_dump:
            fp.write(key + "_*_*_" + value + "\n")
        fp.close()
        self.doc_id_dump_count += 1
        self.doc_id_dump.clear()


    def add_to_index(self, data, pos, document_id):
        for item in data:
            if item in self.invertedIndex[pos]:
                self.invertedIndex[pos][item].append((document_id, data[item]))
            else:
                self.invertedIndex[pos][item] = []
                self.invertedIndex[pos][item].append((document_id, data[item]))
        
    def dump_data(self):
        ##dump data
        idx = 0
        for item in self.mapping:
            dump_file = open(str(self.file_number) + '_' + item + ".txt" , 'w+')
            for key in sorted(self.invertedIndex[idx].keys()):
                dump_file.write(str(key) + "_*_*_")
                for (doc_id, freq) in self.invertedIndex[idx][key]:
                    dump_file.write(" " + str(doc_id) + " " + str(freq))
                dump_file.write('\n')
            dump_file.close()    
            idx += 1
            
        self.invertedIndex.clear()
        for i in range(6):
            self.invertedIndex.append({})
        self.file_number += 1

    def merge_sort(self):
        idx_mapping = 0
        for item in self.mapping:
            file_ptr = []
            for file_n in range(self.file_number):
                dump_file = open(str(file_n) + '_' + item + ".txt" , 'r')
                file_ptr.append(dump_file)
            
            merged_file = open(item + str(self.merge_file_num) + "_inverted_index.txt", 'w+')

            min_heap = []
            idx = 0
            for file_p in file_ptr:
                ## line , file number
                line = file_p.readline()
                line = line.split("_*_*_")
                if(len(line) > 1):
                    key = line[0]
                    value = []
                    nline = line[1].strip().split()
                    for i in nline:
                        value.append(i)

                    min_heap.append((key, idx , value))
                idx += 1

            heapq.heapify(min_heap)
            prev_key = ""
            prev_value = []
            while(len(min_heap) > 0):
                (key, idx, value) = heapq.heappop(min_heap)
                if key == prev_key:
                    prev_value.extend(value)
                else:
                    if(self.new_file):
                        self.level2_index[idx_mapping].append(key)
                        self.new_file = False
                    
                    merged_file.write(prev_key + "_*_*_")
                    for valss in prev_value:
                        merged_file.write(' ' + str(valss))
                    merged_file.write('\n')

                    self.merge_doc_num += 1

                    if(self.merge_doc_num % self.merge_threshold == 0):
                        merged_file.close()
                        self.merge_file_num += 1
                        merged_file = open(item + str(self.merge_file_num) + "_inverted_index.txt", 'w+')
                        self.new_file = True

                    prev_key = key
                    prev_value = value
                    
                line = file_ptr[idx].readline()
                if(len(line) > 0):
                    line = line.split("_*_*_")
                    
                    key = line[0]
                    value = []
                    nline = line[1].strip().split()
                    for i in nline:
                        value.append(i)
                        
                    heapq.heappush(min_heap, (key, idx, value))

            
            for file_p in file_ptr:
                file_p.close()
            
            merged_file.close()
            self.merge_file_num = 0
            idx_mapping += 1

    def dump_level2_index(self):
        for i in range(6):
            level2 = open(self.mapping[i] + "_level2.txt", 'w+')
            for word in self.level2_index[i]:
                level2.write(word + '\n')
            level2.close() 

    def process(self, page):
        page.title = page.title.lower()
        page.text = page.text.lower()
        tokenised_title = self.tokenize(page.title)
        filtered_title = self.removeStopwords(tokenised_title)
        stemmed_title = self.stem_words(filtered_title)
        stemmed_title = self.compress(stemmed_title)

        body, category, infobox, references, ext_links = self.process_text(page.text)
        body = self.compress(body)
        category = self.compress(category)
        infobox = self.compress(infobox)
        references = self.compress(references)
        ext_links = self.compress(ext_links)

        ## make inverted index

        self.add_to_index(stemmed_title, 0 , self.document_id)
        self.add_to_index(body, 1 , self.document_id)
        self.add_to_index(category, 2 , self.document_id)
        self.add_to_index(infobox, 3 , self.document_id)
        self.add_to_index(references, 4 , self.document_id)
        self.add_to_index(ext_links, 5 , self.document_id)
        
        self.doc_id_dump.append((str(self.document_id),page.title ))
        
        self.document_id += 1
        
        if(self.document_id % self.threshold == 0):
            self.dump_data()
        
        if(self.document_id % self.doc_id_threshold == 0):
            self.dump_doc_id_doc_name()


    def removeStopwords(self, text):
        stop_words = set(stopwords.words('english'))
        filtered_word = [word for word in text if not word in stop_words]
        return filtered_word
    
    def stem_words(self, text):
        porter_stemmer = PorterStemmer()
        stemmed_list = []
        for word in text:
            stemmed_list.append(porter_stemmer.stem(word))
        return stemmed_list
    
#     Title (done), Infobox (done), Body(done), Category(done), Links, and
# References of a Wikipedia page.

    def process_text(self, text):
        # topic_list = ["==references==", "==external links==", "[[category"]
        infobox = []
        body = []
        body_flag = True
        ref_flag = False
        category = []
        references = []

        lines = text.split("\n")
        no_of_lines = len(lines)
        for i in range(no_of_lines):
            ## infobox
            if("==references==" in lines[i]):
                ref_flag = True
                continue
            if(ref_flag == True):
                if(len(lines[i].strip()) == 0):
                    ref_flag = False
                else:
                    references.append(lines[i])
                
            if("{{infobox" in lines[i]):
                count = 0
                start = lines[i].split("{{infobox")[1:]
                infobox.extend(start)
                while(True):
                    if("{{" in lines[i]):
                        count += lines[i].count("{{")
                    elif("}}" in lines[i]):
                        count -= lines[i].count("}}")
                    
                    if(count <= 0):
                        break
                    i += 1
                    if(i < len(lines)):
                        break

                    infobox.append(lines[i])

            ## body  
            if(body_flag):
                if("[[category" in lines[i] or "==external links==" in lines[i]):
                    body_flag = False
                else:
                    body.append(lines[i])
            
            else:
                if("[[category" in lines[i]):
                    cat_line = lines[i].split("[[category")
                    for j in range (len(cat_line)):
                        if(j == (len(cat_line)-1)):
                            cat_new = cat_line[j].split("]]")
                            category.append(cat_new[0])
                        else:
                            category.append(cat_line[j])
                    
        category = " ".join(category)
        category = self.tokenize(category)
        category = self.stem_words(category)

        infobox = " ".join(infobox)
        infobox = self.tokenize(infobox)
        infobox = self.stem_words(infobox)

        body = " ".join(body)
        body = self.tokenize(body)
        body = self.stem_words(body)

        ext_links = self.external_links(text)
        
        return body, category, infobox,references, ext_links


    def external_links(self, text):
        ext_links=[]
        data = text
        lines = data.split("==external links==")
        
        if len(lines)>1:
            lines=lines[1].split("\n")
            for i in range(len(lines)):
                if '* [' in lines[i] or '*[' in lines[i]:
                    link = lines[i].split(']')
                    ext_links.append(link[0])
        
        return ext_links



class wikiHandler(xml.sax.ContentHandler):
    def __init__(self):
        self.CurrentData = ""
        self.processor = Processor()
        self.textSetter = False

    def startElement(self, tag, attributes):
        self.CurrentData = tag
        if(tag == "page"):
            self.page = Page()
        if(tag == "text"):
            self.textSetter = True

    def endElement(self, tag):
        if(tag == "page"):
            return self.processor.process(self.page)
            

        self.CurrentData = ""
        
    def characters(self, content):
        if self.CurrentData == "title":
            self.page.title += content
        elif self.CurrentData == "text":
            self.page.text += content

    def endDocument(self):
        self.processor.dump_data()
        ## merge files
        self.processor.merge_sort()
        self.processor.dump_level2_index()
        file_ptr = open("document_count.txt", "w+")
        file_ptr.write(str(self.processor.document_id))
        file_ptr.close()
        self.processor.dump_doc_id_doc_name()
        print("Inverted index created.")
        


if __name__ == "__main__":
    parser = xml.sax.make_parser()
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    Handler = wikiHandler()
    parser.setContentHandler(Handler)
    parser.parse("testing.xml")
