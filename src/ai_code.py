import os
import pandas as pd
import pathlib
import logging
import requests
import feedparser
from typing import Tuple, List, Dict, Set
from collections import defaultdict
import sys
import logging
import feedparser
import pandas as pd
from typing import List, Dict, Tuple
from multiprocessing import Pool
import math
import inspect

logger = logging.getLogger(__name__)

def log_info_with_line_number(message):
    caller_frame = inspect.currentframe().f_back
    caller_module = inspect.getmodule(caller_frame)
    line_number = caller_frame.f_lineno
    logger.info(f"[{caller_module.__name__}:{line_number}] {message}")



logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.absolute()
MAX_ARTICLES_PER_REQUEST = 10


class FileLoader:
    def __init__(self, project_root: pathlib.Path, data_dir: str):
        self.project_root = project_root
        self.data_dir = data_dir

    def get_data(self, filename: str, dtype: dict = None) -> pd.DataFrame:
        full_path = os.path.join(self.project_root, self.data_dir, filename)
        if os.path.exists(full_path):
            existing_data = pd.read_csv(full_path, dtype=dtype)
        else:
            existing_data = pd.DataFrame()

        logging.debug(f"Loading data from {full_path}")
        return existing_data

    def insert_or_update_data(self, df: pd.DataFrame, filename: str) -> None:
        full_path = os.path.join(self.project_root, self.data_dir, filename)
        df.to_csv(full_path, index=False)
        logging.debug(f"Data saved to {full_path}")


class DataManager:
    def __init__(self, project_root: pathlib.Path = PROJECT_ROOT, data_dir: str = 'data', articles_database: str = 'articles.csv', citations_database: str = 'citations.csv'):
        self.project_root = project_root
        self.data_dir = data_dir
        self.articles_database = articles_database
        self.citations_database = citations_database
        self.file_loader = FileLoader(project_root, data_dir)


    def get_data(self, filename: str, dtype: dict = None) -> pd.DataFrame:
        return self.file_loader.get_data(filename, dtype={'arxiv_id': str, 'citing_paper': str, 'cited_paper': str})

    def insert_or_update_data(self, df: pd.DataFrame, filename: str) -> None:
        self.file_loader.insert_or_update_data(df, filename)

    def count_articles_by_processing_status(self) -> pd.DataFrame:
        articles_df = self.get_data(self.articles_database)
        count = articles_df.groupby('processing_status').size().reset_index(name='Count')
        total_count = articles_df.shape[0]
        count = pd.concat([count, pd.DataFrame({'processing_status': ['Total'], 'Count': [total_count]})],
                          ignore_index=True)
        log_info_with_line_number(f"Count of articles by processing status: \n{count}")
        return count

    def percentage_of_processed_articles(self) -> float:
        articles_df = self.get_data(self.articles_database)
        total_articles = len(articles_df)
        processed_articles = articles_df[articles_df['processing_status'] == 'processed']
        percentage = len(processed_articles) / total_articles * 100
        log_info_with_line_number(f"Percentage of processed articles: {percentage}")
        return percentage

    # calculate number of references in citations.csv
    def count_total_references(self) -> int:
        citations_df = self.get_data(self.citations_database)
        count = len(citations_df)
        log_info_with_line_number(f"Number of total references: {count}")
        return count


    def count_unique_citing_and_cited_papers(self) -> int:
        citations_df = self.get_data(self.citations_database)
        citing_papers = set(citations_df['citing_paper'])
        cited_papers = set(citations_df['cited_paper'])
        count = len(citing_papers.union(cited_papers))
        log_info_with_line_number(f"Number of unique citing and cited papers: {count}")
        return count

    def find_citations_not_in_articles(self) -> Tuple[pd.DataFrame, int]:
        articles_df = self.get_data(self.articles_database)
        citations_df = self.get_data(self.citations_database)
        existing_articles = set(articles_df['arxiv_id'])
        citing_papers = set(citations_df['citing_paper'])
        cited_papers = set(citations_df['cited_paper'])
        union_papers = citing_papers.union(cited_papers)
        missing_citations = citations_df[
            ~citations_df['cited_paper'].isin(existing_articles) & citations_df['citing_paper'].isin(union_papers)]
        count = len(missing_citations)
        log_info_with_line_number(f"Number of citations not in articles: {count}")
        return missing_citations, count

    def cited_papers_frequency_table_from_citations(self, min_citations: int = 0) -> pd.DataFrame:
        citations_df = self.get_data(self.citations_database)
        freq_table = citations_df['cited_paper'].value_counts().reset_index()
        freq_table.columns = ['ID', 'Citation Count']
        filtered_freq_table = freq_table[freq_table['Citation Count'] > min_citations]
        log_info_with_line_number(f"Citations frequency table: \n{filtered_freq_table}")
        return filtered_freq_table


class ADSFetcher:
    BASE_URL = 'https://api.adsabs.harvard.edu/v1/search/query?q='
    ROWS = 2000

    def __init__(self, api_key: str):
        self.headers = {
            'Authorization': f'Bearer {api_key}',
        }

    def get_paper_bibcode_references_citations(self, arxiv_id: str) -> Tuple[dict, dict]:
        bibcode = self.get_bibcode(arxiv_id)
        references = self.get_references(bibcode)
        citations = self.get_citations(bibcode)
        return bibcode, references, citations

    def get_bibcode(self, arxiv_id: str) -> str:
        url = f'{self.BASE_URL}identifier:arXiv:{arxiv_id}&fl=bibcode'
        response = self._make_request(url)
        return self._parse_bibcode(response)

    def get_references(self, bibcode: str) -> dict:
        url = f'{self.BASE_URL}references(bibcode:{bibcode})&fl=identifier&rows={self.ROWS}'
        response = self._make_request(url)
        return self._parse_identifiers(response)

    def get_citations(self, bibcode: str) -> dict:
        url = f'{self.BASE_URL}citations(bibcode:{bibcode})&fl=identifier&rows={self.ROWS}'
        response = self._make_request(url)
        return self._parse_identifiers(response)

    def _make_request(self, url: str) -> dict:
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            log_info_with_line_number(f"Request failed: {e}")
            return {}

    def _parse_bibcode(self, response: dict) -> str:
        try:
            return response['response']['docs'][0]['bibcode']
        except KeyError:
            log_info_with_line_number("No bibcode found")
            return ''

    def _parse_identifiers(self, response: dict) -> dict:
        identifiers = [doc['identifier'] for doc in response.get('response', {}).get('docs', [])]
        return self._organize_identifiers(identifiers)

    def _organize_identifiers(self, identifiers: list) -> dict:
        result_dict = defaultdict(list)
        for identifier_list in identifiers:
            for identifier in identifier_list:
                if 'arXiv:' in identifier:
                    result_dict['arxiv'].append(identifier.split('arXiv:')[1])
                elif '10.48550/' in identifier:
                    result_dict['doi'].append(identifier.split('10.48550/')[1])
                else:
                    result_dict['bibcode'].append(identifier)
        return result_dict


class ArxivFetcher:
    BASE_URL = 'http://export.arxiv.org/api/query?'

    def __init__(self, data_loader: DataManager):
        self.data_loader = data_loader
        self.logger = self.setup_logger()

    @staticmethod
    def setup_logger() -> logging.Logger:
        logger = logging.getLogger(__name__)
        logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
        return logger

    def get_data_from_arxiv(self, url: str) -> pd.DataFrame:
        self.logger.debug("Fetching data from Arxiv API")
        feed = feedparser.parse(url)
        data = self.extract_feed_data(feed)
        return pd.DataFrame(data)

    @staticmethod
    def separate_arxiv_id_version(arxiv_id: str) -> Tuple[str, str]:
        if 'v' in arxiv_id:
            arxiv_id, version = arxiv_id.split('v', 1)
            version = 'v' + version
        else:
            version = ''
        return str(arxiv_id), version


    def get_arxiv_data_for_ids(self, arxiv_ids: Set[str]) -> pd.DataFrame:
        logging.debug("Fetching arXiv data for IDs")
        data = []
        url = f'{ArxivFetcher.BASE_URL}id_list={",".join([str(id) for id in list(arxiv_ids)][:MAX_ARTICLES_PER_REQUEST])}'
        feed = feedparser.parse(url)

        if feed.entries:
            for entry in feed.entries:
                link_pdf = entry.links[1]['href'] if len(entry.links) > 1 else 'Not specified'
                arxiv_id = entry.id.split('/abs/')[-1]
                arxiv_id, version = ArxivFetcher.separate_arxiv_id_version(arxiv_id)
                data.append({
                    'arxiv_id': str(arxiv_id),
                    'arxiv_id_version': version,
                    'title': entry.title,
                    'authors': ', '.join(author.name for author in entry.authors),
                    'abstract': entry.summary,
                    'publication_date': entry.published,
                    'categories': ', '.join(tag.term for tag in entry.tags),
                    'journal': entry.arxiv_journal_ref if 'arxiv_journal_ref' in entry else 'Not specified',
                    'link': entry.link,
                    'link_pdf': link_pdf,
                    'arxiv_comment': entry.arxiv_comment if 'arxiv_comment' in entry else ''
                })
            else:
                logging.warning(f"No article found with arXiv ID: {arxiv_id}")

        return pd.DataFrame(data)


    @staticmethod
    def extract_feed_data(feed) -> List[Dict]:
        data = []
        for entry in feed.entries:
            link_pdf = entry.links[1]['href'] if len(entry.links) > 1 else 'Not specified'
            arxiv_id = entry.id.split('/abs/')[-1]
            arxiv_id, version = ArxivFetcher.separate_arxiv_id_version(arxiv_id)
            data.append({
                'arxiv_id': str(arxiv_id),
                'arxiv_id_version': version,
                'title': entry.title,
                'authors': ', '.join(author.name for author in entry.authors),
                'abstract': entry.summary,
                'publication_date': entry.published,
                'categories': ', '.join(tag.term for tag in entry.tags),
                'journal': entry.arxiv_journal_ref if 'arxiv_journal_ref' in entry else 'Not specified',
                'link': entry.link,
                'link_pdf': link_pdf,
                'arxiv_comment': entry.arxiv_comment if 'arxiv_comment' in entry else ''
            })
        return data

    def get_recent_articles_for(self, search_query: str, max_results: int = 100) -> pd.DataFrame:
        logging.debug(f"Fetching recent articles for {search_query}")
        query = f'search_query={search_query}&sortBy=lastUpdatedDate&sortOrder=descending&max_results={max_results}'
        df = self.get_data_from_arxiv(self.BASE_URL + query)
        return df

    @staticmethod
    def add_pending_fields(df: pd.DataFrame) -> pd.DataFrame:
        df['processing_status'] = 'pending'
        df['bibcode'] = ''
        df['num_of_references'] = ''
        df['num_of_citations'] = ''
        return df

    def fetch_and_save(self, search_query: str, filename: str, max_results: int = 100) -> None:
        logging.debug(f"Fetching data for {search_query}")
        existing_data = self.data_loader.get_data(filename)

        if existing_data is not None and not existing_data.empty:
            # Filter the newly downloaded data to exclude records that already exist
            existing_ids = set(existing_data['arxiv_id'])
            df = self.get_recent_articles_for(search_query, max_results)
            new_records = df[~df['arxiv_id'].isin(existing_ids)]
            log_info_with_line_number(f"{len(new_records)} new records found")
            new_records = ArxivFetcher.add_pending_fields(new_records)
            # Merge the new records with the existing data
            df = pd.concat([existing_data, new_records], ignore_index=True)
        else:
            # No existing data, save the downloaded data as is
            log_info_with_line_number("No existing data found")
            df = self.get_recent_articles_for(search_query, max_results)
            df = ArxivFetcher.add_pending_fields(df)

        self.data_loader.insert_or_update_data(df, filename)
        self.logger.debug("Data saved to %s", filename)


class ReferenceProcessor:
    def __init__(self, ads_fetcher: ADSFetcher, data_loader: DataManager):
        self.ads_fetcher = ads_fetcher
        self.data_loader = data_loader

    def get_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logging.debug("Loading articles and citations data")
        articles_df = self.data_loader.get_data('articles.csv')
        citations_df = self.data_loader.get_data('citations.csv')
        return articles_df, citations_df

    def insert_or_update_data(self, articles_df: pd.DataFrame, citations_df: pd.DataFrame) -> None:
        log_info_with_line_number("Saving articles and citations data")
        self.data_loader.insert_or_update_data(articles_df, 'articles.csv')
        self.data_loader.insert_or_update_data(citations_df, 'citations.csv')

    def get_references_and_citations_for_pending_articles(self, fetch_references_of_n: int = None) -> None:
        """Fetch references of articles

        Args:
        process_only_n (int): Maximum number of articles to process. If None, all articles are processed.
        """

        log_info_with_line_number("Fetching references and citations of pending articles")
        try:
            articles_df, citations_df = self.get_data()
            articles_to_process_df = self.filter_pending_articles(articles_df)

            if not articles_to_process_df.empty:
                articles_to_process_df = articles_to_process_df.head(fetch_references_of_n)
                log_info_with_line_number(f"Processing {len(articles_to_process_df)} articles")
                self.process_and_update_articles_and_citations(articles_to_process_df, articles_df, citations_df)
        except Exception as e:
            logging.error(f"Failed to fetch references of articles: {str(e)}")


    def process_and_update_articles_and_citations(self, articles_to_process_df: pd.DataFrame, articles_df: pd.DataFrame,
                                                  citations_df: pd.DataFrame) -> None:
        """Process and save articles

        Args:
        articles_to_process_df (pd.DataFrame): DataFrame of articles to process.
        articles_df (pd.DataFrame): DataFrame of all articles.
        citations_df (pd.DataFrame): DataFrame of all citations.
        """

        articles_to_process_df, citations_df = self.update_articles_and_citations(articles_to_process_df, citations_df)
        # update the articles database with the number of references and citations
        articles_df.update(articles_to_process_df)
        self.insert_or_update_data(articles_df, citations_df)

    def update_articles_and_citations(self, articles_df: pd.DataFrame, citations_df: pd.DataFrame) -> None:
        for index, article in articles_df.iterrows():
            arxiv_id = article['arxiv_id']
            publication_date = article['publication_date']
            log_info_with_line_number(f"Processing article {arxiv_id}")
            # todo: maybe ads fetcher has options to mutliple arxiv ids
            bibcode, references, citations = self.ads_fetcher.get_paper_bibcode_references_citations(arxiv_id)
            references = references['arxiv']
            citations = citations['arxiv']
            log_info_with_line_number(f"Found {len(references)} references and {len(citations)} citations for {arxiv_id}")
            # update the articles database with the number of references and citations
            self.update_article_database_with_num_of_references_and_citations(articles_df, index, bibcode, references, citations)
            # set processing status of article
            if len(references) == 0:
                self.set_processing_status_of_articles(articles_df, index, 'references not found')
            else:
                self.set_processing_status_of_articles(articles_df, index, 'processed')
            # add citations to database
            citations_df = self.add_new_references_and_citations_to_citations_database(citations_df, arxiv_id, publication_date, references, citations)
        return articles_df, citations_df
    def multi_process_articles(self, articles_df: pd.DataFrame, citations_df: pd.DataFrame) -> None:
        with Pool(processes=4) as pool:  # Number of processes depends on your machine's capabilities.
            # Each process will execute the process_article method with a tuple (row[1], citations_df)
            pool.map(self.NOTUSED_process_article, [(row[1], citations_df) for row in articles_df.iterrows()])

    def NOTUSED_process_article(self, data) -> None:
        article, citations_df = data

        arxiv_id = article['arxiv_id']
        publication_date = article['publication_date']
        bibcode, references, citations = self.ads_fetcher.get_paper_bibcode_references_citations(arxiv_id)

        self.update_article_database_with_num_of_references_and_citations(article, bibcode, references, citations)

        if len(references) == 0:
            self.set_processing_status_of_articles(article, 'references not found')
        else:
            self.set_processing_status_of_articles(article, 'processed')

        self.add_new_references_and_citations_to_citations_database(citations_df, arxiv_id, publication_date, references['arxiv'], citations['arxiv'])



    def filter_pending_articles(self, articles_df: pd.DataFrame) -> pd.DataFrame:
        return articles_df[articles_df['processing_status'] == 'pending']

    @staticmethod
    def get_citation_counts(citations_df: pd.DataFrame) -> pd.Series:
        return citations_df['cited_paper'].value_counts()

    @staticmethod
    def order_articles_by_citation_count(articles_df: pd.DataFrame, citation_counts: pd.Series,
                                         process_only_n: int) -> pd.DataFrame:
        articles_df = articles_df.copy()
        articles_df['citation_count'] = articles_df['arxiv_id'].map(citation_counts)

        articles_df.sort_values(by='citation_count', ascending=False, inplace=True)
        if process_only_n is not None:
            articles_df = articles_df.head(process_only_n)
        return articles_df


    def update_article_database_with_num_of_references_and_citations(
            self, articles_df: pd.DataFrame, index: int,
            bibcode: str, references: dict, citations: dict) -> None:
        articles_df.loc[index, 'bibcode'] = bibcode
        articles_df.loc[index, 'num_of_references'] = len(references)
        articles_df.loc[index, 'num_of_citations'] = len(citations)

    def set_processing_status_of_articles(self, articles_df: pd.DataFrame, index: int, status: str) -> None:
        articles_df.loc[index, 'processing_status'] = status

    def create_references_dataframe(self, arxiv_id: str, references: dict, publication_date: str) -> pd.DataFrame:
        references_df = pd.DataFrame({
            'citing_paper': arxiv_id,
            'cited_paper': references,
            'date_of_reference': publication_date
        })
        return references_df

    def add_new_references_and_citations_to_citations_database(self, citations_df: pd.DataFrame,
                                                               arxiv_id: str, publication_date, references: List[str], citations: List[str]) -> None:
        citations_df_temp = citations_df.copy()

        citing_paper_list = [arxiv_id] * len(references) + [citation for citation in citations]
        cited_paper_list = references + [arxiv_id] * len(citations)
        date_list = [publication_date] * (len(references) + len(citations))

        references_df = pd.DataFrame({
            'citing_paper': citing_paper_list,
            'cited_paper': cited_paper_list,
            'date_of_reference': date_list
        })
        merged_citations_df = pd.concat([citations_df_temp, references_df], ignore_index=True)
        merged_citations_df.drop_duplicates(inplace=True)
        log_info_with_line_number(f"Total number of references and citations: {len(citations_df)}")
        return merged_citations_df


class ArticleInjector:
    def __init__(self, data_loader: DataManager, ads_fetcher: ADSFetcher, article_fetcher: ArxivFetcher):
        self.data_loader = data_loader
        self.ads_fetcher = ads_fetcher
        self.article_fetcher = article_fetcher

    def get_new_articles_data_based_on_citations(self, citations_database: str, articles_database: str, inject_highest_cited_n: int = 30) -> None:
        log_info_with_line_number("Fetching data for new articles based on citations")
        df_articles = self.data_loader.get_data(articles_database)
        citations_df = self.data_loader.get_data(citations_database)

        citations_arxiv_ids = self.extract_citations_arxiv_ids(citations_df)
        new_arxiv_ids = list(self.get_new_arxiv_ids(df_articles, citations_arxiv_ids))
        # we filter for those which has the highest cited paper in the database
        citations_df_filtered = citations_df[citations_df['cited_paper'].isin(new_arxiv_ids)]
        citation_counts = ReferenceProcessor.get_citation_counts(citations_df_filtered)
        top_n_indexes = citation_counts.nlargest(inject_highest_cited_n).index
        log_info_with_line_number(f"From citations database the articles with highest citation are:\n{citation_counts.nlargest(inject_highest_cited_n)}")
        new_arxiv_ids = list(top_n_indexes)
        log_info_with_line_number(f"Number of new articles to fetch: {len(new_arxiv_ids)}")

        if len(new_arxiv_ids)>0:
            df_total_pending_articles = pd.DataFrame()  # Initialize an empty DataFrame
            num_requests = math.ceil(len(new_arxiv_ids) / MAX_ARTICLES_PER_REQUEST)

            for i in range(num_requests):
                start_idx = i * MAX_ARTICLES_PER_REQUEST
                end_idx = start_idx + MAX_ARTICLES_PER_REQUEST
                arxiv_ids_batch = new_arxiv_ids[start_idx:end_idx]
                log_info_with_line_number(f"Fetching articles {start_idx} to {end_idx}")
                df_pending_articles = self.article_fetcher.get_arxiv_data_for_ids(arxiv_ids_batch)
                df_pending_articles = ArxivFetcher.add_pending_fields(df_pending_articles)

                df_total_pending_articles = pd.concat([df_total_pending_articles, df_pending_articles], ignore_index=True)

            updated_articles_df = self.merge_pending_with_existing_articles(df_articles, df_total_pending_articles)

            self.data_loader.insert_or_update_data(updated_articles_df, articles_database)

            log_info_with_line_number(f"New articles were added: {len(df_total_pending_articles)}")

    def extract_citations_arxiv_ids(self, citations_df: pd.DataFrame) -> set:
        if citations_df.empty:
            return set()
        return set(citations_df['citing_paper']).union(set(citations_df['cited_paper']))

    def get_new_arxiv_ids(self, cleaned_articles_df: pd.DataFrame, citations_arxiv_ids: set) -> set:
        existing_arxiv_ids = set(cleaned_articles_df['arxiv_id'])
        return citations_arxiv_ids.difference(existing_arxiv_ids)

    def merge_pending_with_existing_articles(self, cleaned_articles_df: pd.DataFrame,
                                             pending_articles: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([cleaned_articles_df, pending_articles], ignore_index=True)


class ArticleProcessing:
    def __init__(self):


        self.data_loader = DataManager()
        self.article_fetcher = ArxivFetcher(self.data_loader)
        self.ads_fetcher = ADSFetcher(os.environ["ADS_DEV_KEY"])
        self.reference_processor = ReferenceProcessor(self.ads_fetcher, self.data_loader)
        self.article_injector = ArticleInjector(self.data_loader, self.ads_fetcher, self.article_fetcher)

    def get_initial_papers(self, max_results: int = 10):
        self.article_fetcher.fetch_and_save('cat:cs.CL', 'articles.csv', max_results=max_results)
        self.data_loader.count_articles_by_processing_status()

    def process_article_data(self, process_cycles: int = 3, fetch_references_of_n: int = 2, inject_highest_cited_n: int = 2):
        for i in range(process_cycles):
            log_info_with_line_number(f"Processing cycle {i + 1}")
            # get pending articles
            self.reference_processor.get_references_and_citations_for_pending_articles(fetch_references_of_n=fetch_references_of_n)

            # get references and citations for pending articles
            self.article_injector.get_new_articles_data_based_on_citations('citations.csv', 'articles.csv', inject_highest_cited_n=inject_highest_cited_n)

            # diagnostics
            self.data_loader.count_articles_by_processing_status()
            self.data_loader.find_citations_not_in_articles()

    # Get the processed citations
    # def get_processed_citations(self, citations_query: dict):
    #     return self.data_loader.process_citations_csv(citations_query)


def main():
    article_processing = ArticleProcessing()

    if True:
        article_processing.get_initial_papers(10)

    article_processing.process_article_data(process_cycles=3, fetch_references_of_n=100, inject_highest_cited_n=10)

    # Get the processed citations
    # citations_query = {'citing_paper': 'Paper 1', 'cited_paper': 'Paper A'}
    # citations_df = article_processing.get_processed_citations(citations_query)


if __name__ == '__main__':
    main()

