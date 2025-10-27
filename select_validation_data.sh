cd /data/group_data/cx_group/WebOrganizer/Corpus-30B/documents
ls CC_shard_*.jsonl | tail -20 | xargs -I {} cp {} ../selected-1B/validation_documents/