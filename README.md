# Wikipedia-Search
TF-IDF based Search Engine to search in Wikipedia Data dump.

Run wiki.py to form the inverted index and other necessery files for performing query.

    -> python3 wiki.py

After running wiki.py, we will use search.py to perform search query

    -> python3 search.py

Each query takes 2 inputs:

    1) field in which to perform search or in all fields. it is one of these values : 
        ['title', 'body' , 'category', 'infobox' , 'references', 'ext_links']
        and 'not_field' for performing search in all the fields.
        
    2) query string

Result of each query is document_id and Document Name ranked by TF-IDF. A max of 10 documents will be
present in the result.

enter 'exit' while entering field in query to quit.
