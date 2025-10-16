import argparse
from app.services.embedding_service import reindex_documents


def run(directory=None):
	reindex_documents(directory)
	print("✅ Indexing complete and persisted to Chroma.")


def main():
	parser = argparse.ArgumentParser(description="Index documents into Chroma using LlamaIndex")
	parser.add_argument("--dir", dest="directory", default=None, help="Directory of documents to index")
	args = parser.parse_args()
	run(args.directory)


if __name__ == "__main__":
	main()
