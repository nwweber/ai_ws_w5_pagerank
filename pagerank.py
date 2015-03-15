import hashlib
import os
import queue
import re
import shutil
import urllib.request
from bs4 import BeautifulSoup
import numpy as np
import time
import sys
import pickle
import matplotlib.pyplot as plt

__author__ = 'niklas'


class UniqueQueue(queue.Queue):
    """
    A set-based queue. When adding an item that is already on the queue, no duplicate is added.
    When popping, no fixed order is guaranteed.
    """
    def _init(self, maxsize):
        self.queue = set()

    def _put(self, item):
        self.queue.add(item)

    def _get(self):
        return self.queue.pop()


def scrape(baseurl):
    """
    Scrape from URL
    Follow links in to URL/*
    Generate directed connectivity matrix

    Idea:
    Start at URL
    Download document
    Find all links
    For each link:
        (if new link: put on visiting queue)

        If pointing to URL/*
        and not visited
        and not on queue
            Put on queue

        Save destination of outgoing links

    Repeat for each link on queue till queue empty

    Build matrix from gathered connectivity data:
        M[i][j] : 1 iff there is link going from i to j

    Data structures:
    baseurl = string URL
    doc = current document, BeautifulSoup object
    pages =

    Pseudo:

    q = UniqueQueue()
    visited = Set()
    doc = None
    reg_exp = RegExp(baseurl+".*")
    from_to = {}

    q.put(baseurl)

    while not q.isEmpty():
        url = q.pop()
        fname = download(url)
        doc = parse(fname)
        links = find_all(doc, "a")
        links_to = Set()
        for link in links:
            target = link.href
            if not reg_exp.match(target):
                continue
            if not target in visited:
                q.put(target)
            links_to.add(target)
        visited.add(url)
        from_to[url] = links_to

    all_sites = list(visited)
    n_sites = len(all_sites)

    M = zeros((n_sites, n_sites))
    for i in range(n_sites):
        for j in range(n_sites):
            origin = all_sites[i]
            dest = all_sites[j]
            M[i][j] = int(dest in from_to[origin])

    return (all_sites, M)

    :param baseurl:
    :return: list of all sites, M[i][j] = 1 iff there is a link from page i to page j
    """

    q = UniqueQueue()
    visited = set()
    doc = None
    reg_exp_pattern = baseurl+".*"
    reg_exp = re.compile(reg_exp_pattern)
    from_to = {}

    q.put(baseurl)

    start = time.time()
    counter = 0

    while not q.empty():
        counter += 1
        if counter % 10 == 0:
            diff = time.time() - start
            print("processed: ", counter, "sites in ", diff, "seconds")
            print("processing at ", counter/diff, "sites/second")
        url = q.get()
        visited.add(url)
        print("queue not empty, now processing ", url)
        print("current length of queue: ", q.qsize())
        links_to = set()
        storage_dir = os.path.join(".", "scraped")
        fname = os.path.join(storage_dir, hashlib.md5(bytes(url, sys.getdefaultencoding())).hexdigest())
        if not os.path.isfile(fname):
            print("fname is file? ", fname, os.path.isfile(fname))
            t = 0.3
            print("sleeping for ", t, "seconds")
            time.sleep(t)
            print("scraping from ", url, "to ", fname)
            file_name_temp, headers = urllib.request.urlretrieve(url)
            shutil.copy(file_name_temp, fname)
        else:
            print("url ", url, "already downloaded, using that file instead")
        with open(fname, "r", encoding=sys.getdefaultencoding()) as f:
            html = f.read()
        doc = BeautifulSoup(html)
        links = doc.find_all("a")
        for link in links:
            target = link.get("href")
            # some links are weird and don't have a href attribute. we want to skip those
            if not target:
                continue
            print("link found: ", target)
            if not reg_exp.match(target):
                print("link does not match subdomain")
                continue
            if target not in visited:
                print("link not visited yet, adding to queue")
                q.put(target)
            links_to.add(target)
        from_to[url] = links_to

    all_sites = list(visited)
    n_sites = len(all_sites)

    print("done crawling")
    print("found ", n_sites, "sites")

    m = np.zeros((n_sites, n_sites))
    for i in range(n_sites):
        for j in range(n_sites):
            origin = all_sites[i]
            dest = all_sites[j]
            m[i][j] = int(dest in from_to[origin])

    return all_sites, m


def normalize_rows(X):
    """
    Normalize X row-wise
    :param X: nd-array
    :return: another array where for each row : row.sum() == 1
    """
    row_sums = X.sum(axis=1)
    # The division operation will be broadcast along the rows (axis=0)
    # This is the opposite of what we want. So we transpose X first, then transpose the result back :)
    return (X.T / row_sums).T


def fill_empty_rows(Q_star):
    pass

def fill_and_norrmalize(X):
    row_sums = X.sum(axis=1)
    rsmask = row_sums == 0
    X = (X[rsmask] + 1/X.shape[0]) + (X[~ rsmask].T / row_sums[~ rsmask]).T
    return X

def fill_and_normalize(X):
    """
    Fill empty rows with 1s. Then normalize by row
    :param X: NP array
    :return: Ref to changed array.
    """
    row_sums = X.sum(axis=1)
    mask = row_sums == 0
    X[mask] = 1
    row_sums = X.sum(axis=1)
    X = (X.T / row_sums).T
    return X


def compute_pagerank(Q):
    """
    Approximate the stationary distribution for Q
    :param Q: Normalized socio matrix
    :return: A vector, one element per page in Q, each element being its rank
    """
    J = np.ones(Q.shape)
    m = Q.shape[0]
    alpha = 0.85
    G = alpha * Q + (1 - alpha) * (J / m)
    s = np.random.uniform(0, 1, m)
    s = s / s.sum()
    history = []
    for i in range(50):
        s = np.dot(s, G)
        history.append(s.copy())
    return history


def plot_ranks_history(ranks_history):
    """
    Show how ranks change over time
    :param ranks_history: a list, each element a ndarray vector of ranks at that iteration
    :return:
    """
    h = np.array(ranks_history).T
    for row in h:
        plt.plot(row)
    plt.show()


if __name__ == "__main__":
    crawl = False
    pickle_name = "data.pickle"
    if crawl:
        site_names, conn_matrix = scrape("http://www.ru.nl/artificialintelligence/")
        with open(pickle_name, "wb") as f:
            pickle.dump((site_names, conn_matrix), f)
    else:
        with open(pickle_name, "rb") as f:
            site_names, conn_matrix = pickle.load(f)
    X = conn_matrix
    Q = fill_and_normalize(X)
    ranks_history = compute_pagerank(Q)
    plot_ranks_history(ranks_history)