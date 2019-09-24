import re
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer 
import math
import operator
import time

class SearchWiki:
    def __init__(self):
        self.total_docs = 100
        self.doc_id_threshold = 1000
        print("initiating Searching.")
        self.mapping = ['title', 'body' , 'category', 'infobox' , 'references', 'ext_links']
        self.mapping_idx = {'title':0, 'body':1 , 'category':2, 'infobox':3 , 'references':4, 'ext_links':5}
        self.level2_index = []
        
        for i in range(6):
            self.level2_index.append([])
        
        for idx in range(6):
            file_ptr = open(self.mapping[idx] + "_level2.txt", 'r')
            level2 = file_ptr.readlines()
            for item in level2:
                self.level2_index[idx].append(str(item).replace('\n', ''))
            file_ptr.close()
        
        file_ptr = open("document_count.txt", 'r')
        self.total_docs = int(file_ptr.readline().strip())
        file_ptr.close()
        
        self.document_name = {}
        
        # with open("doc_id_doc_name_mapping.txt", 'r') as fp:
        #     for lines in fp:
        #         line = lines.split("_*_*_")
        #         self.document_name[int(line[0])] = line[1]
        

        # print(self.level2_index)
    
    def tokenize(self, data):                                              #Tokenise
        tokenisedWords=re.findall("\d+|[\w]+",data)
        return tokenisedWords
    
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
    
    def get_file_number(self, word, field):
        field_idx = self.mapping_idx[field]
        idx = 0
        for init_word in self.level2_index[field_idx]:
            if(init_word > word):
                break
            else:
                idx += 1
        
        return idx
        

    def fetch_inverted_index(self, query, field):
        result = []
        IDF = {}
        word_list = {} ## {word, {doc_no, term_freq} }
        for word in query:
            file_no = self.get_file_number(word, field)
            
            with open(field + str(file_no) + "_inverted_index.txt", 'r') as fp:
                for lines in fp:
                    key = ""
                    value = []
                    value.append([])
                    value.append([])    
                    line = lines
                    line = line.split("_*_*_")
                    if(len(line) > 1):
                        key = line[0]
                        if(key == word):
                            nline = line[1].strip().split()
                            for i in range(len(nline)):
                                value[i%2].append(nline[i])

                    if(key == word):
                        IDF[key] = len(value[0])
                        result.append((key, value))
                        if word in word_list:
                            for i in range(len(value[0])):
                                doc_no, term_freq = value[0][i] , value[1][i]
                                word_list[word][doc_no] = term_freq
                        else:
                            word_list[word] = {}
                            for i in range(len(value[0])):
                                doc_no, term_freq = value[0][i] , value[1][i]
                                word_list[word][doc_no] = term_freq

                        break
        
        return result, IDF, word_list
    
    def union(self, TF):
        union_set = set()
        for (key, value) in TF:
            doc_ids = value[0]
            doc_ids = set(doc_ids)
            union_set = union_set.union(doc_ids)
        
        union_set = list(sorted(union_set))
        return union_set
    
    def ranking(self, union_set, word_list, IDF ):
        
        ranks = {} # {doc_id, rank}
        for doc_id in union_set:
            rank = 0.0
            for word in word_list:
                if doc_id in word_list[word]:
                    tf = math.log10(1 + int(word_list[word][doc_id])) ## TODO 
                    idf = math.log10(self.total_docs/int(IDF[word]))
                    rank += tf*idf
            ranks[doc_id] = rank

        return ranks

    def merged_union(self, union_sets):
        merged_union_set = set()
        for sets in union_sets:
            merged_union_set = merged_union_set.union(set(sets))
        merged_union_set = list(sorted(merged_union_set))
        return merged_union_set


if __name__ == "__main__":
    searchwiki = SearchWiki()
    while(True):
        field = input("Enter Field : ")
        if(field == "exit"):
            break
        query = input("Enter Query : ")
        start_time = time.time()
        query = query.lower()
        tokenised_query = searchwiki.tokenize(query)
        filtered_query = searchwiki.removeStopwords(tokenised_query)
        stemmed_query = searchwiki.stem_words(filtered_query)
        print(stemmed_query)
        ranked_pages = {}
        if(field == "not_field"):
            union_sets = []
            IDFs = []
            word_lists = []
            for field in searchwiki.mapping:
                result, IDF, word_list = searchwiki.fetch_inverted_index(stemmed_query,field)
                union_set = searchwiki.union(result)
                union_sets.append(union_set)
                word_lists.append(word_list)
                IDFs.append(IDF)
            
            
            merged_union_set = searchwiki.merged_union(union_sets)
            
            merged_IDF = {}
            for cur_idf in IDFs:
                for key in cur_idf:
                    if key in merged_IDF:
                        merged_IDF[key] += cur_idf[key]
                    else:
                        merged_IDF[key] = cur_idf[key]
            
            merged_word_list = {}
            for cur_word_list in word_lists:
                for words in cur_word_list:
                    if(words in merged_word_list):
                        for cur_doc_id in cur_word_list[words]:
                            if(cur_doc_id in merged_word_list[words]):
                                merged_word_list[words][cur_doc_id] += cur_word_list[words][cur_doc_id]
                            else:
                                merged_word_list[words][cur_doc_id] = cur_word_list[words][cur_doc_id]
                    else:
                        merged_word_list[words] = {}
                        for cur_doc_id in cur_word_list[words]:
                            merged_word_list[words][cur_doc_id] = cur_word_list[words][cur_doc_id]

            ranked_pages = searchwiki.ranking(merged_union_set, merged_word_list, merged_IDF)

        else:
            result, IDF, word_list = searchwiki.fetch_inverted_index(stemmed_query,field)
            union_set = searchwiki.union(result)
            ranked_pages = searchwiki.ranking(union_set, word_list, IDF)


        end_time = time.time()
        print("Time elapsed : " + str(end_time - start_time) + " seconds.")
        sorted_ranked_pages = sorted(ranked_pages.items(), key=operator.itemgetter(1), reverse = True)
        limit = 0
        for (key, value) in sorted_ranked_pages:
            with open(str(int(int(key)/searchwiki.doc_id_threshold)) + "_doc_id_doc_name_mapping.txt", 'r') as fp:
                for lines in fp:
                    line = lines.split("_*_*_")
                    if(str(key) == line[0]):
                        print(line[0] + " - " + line[1])
                        
            limit += 1
            if(limit > 10):
                break
        




