"""Benchmark script for vectorization of user questions """
"""and embeddings lookup"""

import gc
import sys
import threading
from contextlib import contextmanager
from time import time
from statistics import mean
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from llama_index import ServiceContext, StorageContext, load_index_from_storage
from llama_index.core.base_query_engine import BaseQueryEngine
from llama_index.vector_stores.faiss import FaissVectorStore


short = [
    "What is the OLM?",
    "What is OpenShift?",
    "How can I troubleshoot pod deployment failures in OpenShift?",
    "What is the difference between a DeploymentConfig and a Deployment in OpenShift?",
    "How do I scale an application horizontally in OpenShift?",
    "What are Operators in OpenShift, and how do they simplify application management?",
    "Can you explain the process of upgrading OpenShift clusters to a newer version?",
    "How do I configure persistent storage for applications running on OpenShift?",
    "What are the best practices for securing OpenShift clusters and applications?",
    "How can I monitor the health and performance of OpenShift clusters?",
    "What is the recommended approach for managing secrets and sensitive information in OpenShift?",
    "How do I integrate external services, such as databases or message queues, with applications deployed on OpenShift?"
]

medium = [
    "How can I optimize resource utilization and minimize costs for applications running on OpenShift?",
    "What are the implications and best practices for implementing multi-tenancy in OpenShift clusters?",
    "How do I configure custom networking solutions, such as network policies and service mesh, in OpenShift?",
    "Can you explain how to set up high availability and disaster recovery for OpenShift clusters across multiple regions?",
    "What strategies can be employed to efficiently manage and distribute container images within an OpenShift environment?",
    "How can I implement automated deployment pipelines with advanced deployment strategies, such as canary deployments or blue-green deployments, in OpenShift?",
    "What are the considerations and steps involved in integrating OpenShift with external authentication providers, such as LDAP or OAuth?",
    "How can I troubleshoot performance bottlenecks and optimize the performance of applications running on OpenShift?",
    "What is the process for implementing custom resource definitions (CRDs) and operators for managing complex applications in OpenShift?",
    "Can you explain the architecture and components of OpenShift Container Storage (OCS), and how it can be integrated with OpenShift clusters for persistent storage needs?",
]

long = [
    "An application deployed on OpenShift is experiencing intermittent connectivity issues. Despite thorough investigation, the root cause remains elusive. What additional steps can be taken to diagnose and resolve the problem?",
    "OpenShift cluster nodes are reporting high resource utilization, causing performance degradation across applications. How can the cluster be optimized to alleviate resource contention and improve overall performance?",
    "The OpenShift cluster is nearing capacity, and there is a need to expand resources to accommodate growing application workloads. What strategies can be employed to scale the cluster effectively while minimizing downtime?",
    "An unauthorized user gained access to sensitive data stored within an application deployed on OpenShift. How can access controls and security policies be tightened to prevent similar security breaches in the future?",
    "The OpenShift cluster is experiencing frequent outages due to hardware failures in the underlying infrastructure. What measures can be implemented to enhance cluster resilience and minimize downtime?",
    "An application deployed on OpenShift is exhibiting unexpected behavior after a recent update. How can the cluster be rolled back to a previous state to restore stability while investigating the root cause of the issue?",
    "The OpenShift cluster is failing to meet compliance requirements, leading to regulatory concerns and potential penalties. What steps can be taken to ensure that the cluster adheres to regulatory standards and industry best practices?",
    "Persistent storage volumes attached to pods in the OpenShift cluster are becoming corrupted, resulting in data loss and application downtime. How can data integrity be maintained and storage reliability improved within the cluster?",
    "Continuous integration/continuous deployment (CI/CD) pipelines are failing intermittently, causing delays in application delivery. What measures can be implemented to stabilize the CI/CD process and ensure reliable deployment automation?",
    "An application deployed on OpenShift is experiencing a significant increase in traffic, leading to performance degradation and service disruptions. How can the cluster be auto-scaled dynamically to handle fluctuations in workload demand effectively?",
]

questions = [short, medium, long]

results: dict[int, list[int]] = {}
results_lock = threading.Lock()
embeddings_model: HuggingFaceBgeEmbeddings
query_engine: BaseQueryEngine

def thread_func():
    for g in questions:
        for q in g:
            with results_lock:
                r = query_engine.query(q)
                print(r)
                results.setdefault(len(q), []).append(
                    timedreps(10, lambda: query_engine.query(q))
                )


@contextmanager
def gc_disabled():
    """Context manager to temporarily disable garbage collection."""
    try:
        gc.collect()
        gc.disable()
        yield
    finally:
        gc.enable()


def timedreps(reps: int, func: callable) -> float:
    """Run a function reps time, report the average wallclock time for a single call in seconds."""
    start = time()
    for i in range(reps):
        func()
    end = time()
    return int(((end - start) / reps)*1000)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python benchmark-embeddings.py <thread_count> <embeddings_model_dir> <vector_db_dir>")
        sys.exit(1)

    thread_count = int(sys.argv[1])
    embeddings_model_dir = sys.argv[2]
    vector_db_dir = sys.argv[3]

    #print("Loading model...")

    embeddings_model = HuggingFaceBgeEmbeddings(model_name=embeddings_model_dir)
    vector_store = FaissVectorStore.from_persist_dir(vector_db_dir)
    storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=vector_db_dir)
    service_context = ServiceContext.from_defaults(llm=None, embed_model=embeddings_model)
    index = load_index_from_storage(storage_context, index_id="product", service_context=service_context)
    query_engine = index.as_query_engine(response_mode="tree_summarize")


    #print("Model loaded, warming up...")

    # warm up
    thread_func()
    results.clear()

    #print("Warmed up, starting benchmarking run...")

    threads = []

    for t in range(thread_count):
        thread = threading.Thread(target=thread_func)
        threads.append(thread)

    with gc_disabled():
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    # for k, v in results.items():
    #     print(k, mean(v))

    for k in sorted(results.keys()):
        print(k, int(mean(results[k])))
