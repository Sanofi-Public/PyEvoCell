import requests
from xml.etree import ElementTree
import re
import sys
import html
import requests
from xml.etree import ElementTree

def paper_exists(title):
    n_gram = min(8,len(title.split()))
    all_potential_titles = generate_ngrams(title, n_gram)
    already_found_one=False
    for i in all_potential_titles:
        result=check_pubmed_paper(i)
        if result:
            return result

def generate_ngrams(phrase, n):
    words = phrase.split()
    ngrams = []
    for i in range(len(words) - n + 1):
        ngrams.append(' '.join(words[i:i + n]))
    return ngrams

def check_pubmed_paper(title):
    base_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
    search_url = base_url + 'esearch.fcgi'
    
    # Constructing the query parameters
    params = {
        'db': 'pubmed',
        'term': title,
        'retmode': 'xml'
    }
    
    try:
        # Making the request to PubMed API
        attempt = 1
        response = requests.get(search_url, params=params)
        while response.status_code != 200 and attempt < 100:
            response = requests.get(search_url, params=params)
            attempt += 1
        
        # Parse the XML response
        root = ElementTree.fromstring(response.content)
        
        # Extracting the IDs from the response
        id_list = root.find('IdList')
        if id_list is None or len(id_list) == 0:
            return None
        
        # Extract the first ID (if any)
        first_id = id_list.find('Id').text
        
        # Fetch the detailed information for the first ID
        fetch_url = base_url + 'efetch.fcgi'
        fetch_params = {
            'db': 'pubmed',
            'id': first_id,
            'retmode': 'xml'
        }
        fetch_response = requests.get(fetch_url, params=fetch_params)
        fetch_root = ElementTree.fromstring(fetch_response.content)
        
        # Extract the title of the paper
        article_title = fetch_root.find('.//ArticleTitle')
        if article_title is not None:
            article_title = article_title.text
        else:
            article_title = 'No title available'
        
        # Extract the authors
        authors = []
        for author in fetch_root.findall('.//Author'):
            last_name = author.find('LastName')
            fore_name = author.find('ForeName')
            if last_name is not None and fore_name is not None:
                authors.append(f"{fore_name.text} {last_name.text}")
        authors = ', '.join(authors) if authors else 'No authors available'
        
        # Extract the journal
        journal = fetch_root.find('.//Journal/Title')
        if journal is not None:
            journal = journal.text
        else:
            journal = 'No journal available'
        
        # Extract the publication date
        pub_date = fetch_root.find('.//PubDate')
        if pub_date is not None:
            pub_year = pub_date.find('Year')
            pub_month = pub_date.find('Month')
            pub_day = pub_date.find('Day')
            if pub_year is not None and pub_month is not None and pub_day is not None:
                pub_date = f"{pub_year.text} {pub_month.text} {pub_day.text}"
            elif pub_year is not None and pub_month is not None:
                pub_date = f"{pub_year.text} {pub_month.text}"
            elif pub_year is not None:
                pub_date = pub_year.text
            else:
                pub_date = 'No date available'
        else:
            pub_date = 'No date available'
        
        # Construct the citation
        citation = f"{authors}. {article_title}. {journal}. {pub_date}."
        return citation
    
    except Exception as e:
        return None

def process_paragraph(paragraph):
    paragraph = paragraph.replace('<br>', ' ')
    # print("paragraph after processed",paragraph)
    # Regex to match a title followed by a digit
    pattern = r'(Title \d+:|Title\d+:)'
    
    # Split the paragraph using the regex pattern, keeping the initial part
    parts = re.split(pattern, paragraph)
    # print("parts",parts)
    # Combine initial part and title-content pairs
    combined_parts = [parts[0].strip()]
    first_part = combined_parts[0]
    yes = 'yes'
    no = 'no'
    if yes in first_part.lower():
        result = ['Yes, your claim is valid. Here are some papers to support my result.']
    elif no in first_part.lower():
        result = ['No, your claim is not valid. Here are some papers to support my result.']
    else:
        return paragraph
    
    with_backup=False
    for i in range(1, len(parts), 2):
        this_title=parts[i + 1].strip()
        # print("this_title",this_title)
        citation = paper_exists(this_title)
        if citation:
            with_backup=True
            result.append(f'\n\nPubMed citation: "{citation}"')
    if not with_backup:
        result.append('Unfortunately, I have no papers to back it up.')

    processed_paragraph = ''.join(result)
    processed_paragraph = html.escape(processed_paragraph)
    processed_paragraph = processed_paragraph.replace('\n', '<br>')
    return processed_paragraph


if __name__ == "__main__":
    paragraph = sys.argv[1]
    processed_paragraph = process_paragraph(paragraph)
    print(processed_paragraph)
