import os
import queue
import re
import shutil
import urllib.request
from bs4 import BeautifulSoup
import numpy as np
import time

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
        t = 0.3
        print("sleeping for ", t, "seconds")
        time.sleep(t)
        links_to = set()
        storage_dir = os.path.join(".", "scraped")
        fname = os.path.join(storage_dir, str(hash(url)))
        print("scraping from ", url, "to ", fname)
        file_name_temp, headers = urllib.request.urlretrieve(url)
        shutil.copy(file_name_temp, fname)
        with open(fname, "r") as f:
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

if __name__ == "__main__":
    site_names, conn_matrix = scrape("http://www.ru.nl/artificialintelligence/")