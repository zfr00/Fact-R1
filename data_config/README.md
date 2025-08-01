# Fact-R1: Towards Explainable Video Misinformation Detection with Deep Reasoning

## FakeVV Comment Raw Data(All IDs are anonymized.)

data_config/comment_raw_data.txt
This data record contains information about a video comment. Here is the meaning of each field:

- **imp_date**: `20250220` - Important date in the format YYYYMMDD, representing February 20, 2025.
- **ftime**: `2025-02-20 07:00:18.911` - Full timestamp, including milliseconds.
- **extinfo**: `extinfo=9.220.203.134` - Additional information, such as an IP address or data source identifier.
- **c_id**: `UgwXTTAP3ptsZzGV80d4AaABAg` - Comment ID or unique identifier for content within the platform.
- **type**: `youtube_short` - Content type, indicating a YouTube short video.
- **source**: `youtube` - The platform from which the data is sourced.
- **raw_id**: `bbf562ee570a3ddad76464757d4334b5` - Raw identifier, a unique ID for the content or user.
- **mid**: `UC9j4zT1K7Cxys6QsJPkmpRA` - Media ID or channel ID on YouTube.
- **root_c_id**: `UgwXTTAP3ptsZzGV80d4AaABAg` - Root comment ID, indicating the original comment in a thread.
- **parent_c_id**: (empty) - Parent comment ID, indicating the comment this is replying to, if applicable.
- **ext**: A JSON object containing extended metadata or content details, including:
  - **content**: The actual text of the comment.
  - **compete_doc_id**: Document ID for competitive analysis.
  - **crawl_source**: Source of the crawl, `youtube_pc`.
  - **cp_name**: Name of the content provider or user.
  - **report_id**: Report identifier.
  - **like_count**: Number of likes the comment has received.
- **pub_time**: `2024-10-20 00:00:00` - Publication time of the content.
- **created_at**: `2025-02-20 07:00:18` - Timestamp indicating when the record or data entry was created.

## Fact-R1 Pipeline Dataset
Here, we present a portion of the dataset used for the three-stage training process. Please note that the training sets for the third stage, which include fakesv and fakett, are not displayed due to access restrictions.

- data_config/long_cot_random_sampled_data.json
- data_config/dpo_training_data_sampled.json
- data_config/grpo_training_data_sampled.jsonl

