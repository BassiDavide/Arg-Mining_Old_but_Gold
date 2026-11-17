# Dehydrated dataset: YouTube comment chains

This folder contains the train and test splits used in our study, released in **dehydrated** form.
Comment texts and author names have been removed, so the files contain only YouTube IDs, metadata
and our annotations. The script `rehydrate.py` downloads the removed content from the
YouTube Data API v3 and rebuilds the files in their original format.

| File | Contents |
|---|---|
| `train_dataset.json` | Training split: 213 chains, 1,920 messages |
| `test_dataset.json` | Test split: 92 chains, 773 messages |
| `rehydrate.py` | Restores the removed fields |

## Quick start

**Requirements:** Python 3.7 or later and an internet connection. No extra packages are needed.

1. **Get a YouTube Data API v3 key** (free):
   1. Go to the [Google Cloud Console](https://console.cloud.google.com/) and create a project (or select an existing one).
   2. Open *APIs & Services → Library*, search for **YouTube Data API v3** and click **Enable**.
   3. Open *APIs & Services → Credentials*, click **Create credentials → API key** and copy the key.
2. **Add the key** to the top of `rehydrate.py`:
   ```python
   API_KEY = "your-key-here"
   ```
   You can also leave the file unchanged and set an environment variable instead:
   `export YOUTUBE_API_KEY="your-key-here"`.
3. **Run the script** from this folder:
   ```bash
   python rehydrate.py
   ```

The rehydrated files are written to `rehydrated/`:

| Output | Contents |
|---|---|
| `rehydrated/train_dataset.json`, `rehydrated/test_dataset.json` | The full dataset with texts and author names restored |
| `rehydrated/missing_comments.txt` | IDs of comments that could not be retrieved (one per line) |
| `rehydrated/comments_cache.json` | Downloaded comments. Lets a re-run skip the API |

The whole process makes about 51 API requests (2,513 unique comments, 50 per request). Each request
costs 1 quota unit, and the default free quota is 10,000 units per day.

If the run is interrupted (network error, quota), run the script again: it resumes from the cache.
To download everything again, delete `rehydrated/comments_cache.json`.

To use different folders: `python rehydrate.py --in-dir <dehydrated folder> --out-dir <output folder>`.

## What is removed and how it is restored

| Field | Status in the dehydrated files | Restored from |
|---|---|---|
| `CommentText` | removed | the text of the comment `CommentID` |
| `AuthorName` | removed | the author of the comment `CommentID` |
| `ParentCommentText` | removed | the text of the comment `responding_to.comment_id` (empty string for top-level comments) |
| `responding_to.author` | removed | the author of the comment `responding_to.comment_id` |
| chain-level `summary` | removed | not restored; it was not used in the experiments |

All other fields are distributed unchanged. The comment that each message replies to
(`responding_to.comment_id`) is always included in the same chain.

## Data format

Both files share the same structure: a set of conversation chains, each holding an ordered list of messages.

```json
{
  "discussions": {
    "<chain_id>": {
      "messages": [
        {
          "CommentID": "Ugw0oumTBi2PkXQQZhV4AaABAg.9sZ_ivk-NkJ9t4Yt60_17S",
          "ThreadID": "Ugw0oumTBi2PkXQQZhV4AaABAg",
          "VideoID": "2i8n6jCHrS4",
          "ParentCommentID": "Ugw0oumTBi2PkXQQZhV4AaABAg",
          "CommentText": "...",            // restored
          "AuthorName": "...",             // restored
          "NumberOfLikes": 0,
          "IsReply": "True",
          "Timestamp": "2023-08-06T16:37:44Z",
          "Response to": "Ugw0oumTBi2PkXQQZhV4AaABAg.9sZ_ivk-NkJ9t3WLDNS_38",
          "Level": 6,
          "ParentCommentText": "...",      // restored
          "Stance_Label": 1,
          "Techniques": [],
          "responding_to": {
            "author": "...",               // restored
            "comment_id": "Ugw0oumTBi2PkXQQZhV4AaABAg.9sZ_ivk-NkJ9t3WLDNS_38"
          },
          "divisiveness_score": -1,
          "annotation_uncertain": false,
          "incorrect_parent": false,
          "parent_comment_stance": 0
        }
      ]
    }
  }
}
```

| Field | Description |
|---|---|
| `CommentID` | YouTube comment ID |
| `ThreadID` | ID of the top-level comment of the YouTube thread |
| `VideoID` | YouTube video ID (`https://www.youtube.com/watch?v=<VideoID>`) |
| `ParentCommentID` | YouTube parent of the reply, i.e. the top-level comment (empty for top-level comments) |
| `NumberOfLikes`, `Timestamp` | Values at collection time |
| `IsReply` | `"True"` / `"False"` |
| `Response to`, `responding_to.comment_id` | ID of the comment this message actually replies to inside the chain (`null` for the first message of a chain) |
| `Level` | Position in the reply chain (0 = top-level comment) |
| `Stance_Label`, `Techniques`, `divisiveness_score`, `annotation_uncertain`, `incorrect_parent`, `parent_comment_stance` | Annotations (see the paper). The last four fields are not present on top-level comments. |

Some comments appear in more than one chain, and a few appear in both splits, because chains
from the same YouTube thread can share messages.

## Limitations

- **Unavailable comments.** Comments that were deleted, hidden or made private after collection
  cannot be retrieved. Their `CommentText` / `AuthorName` (and the `ParentCommentText` /
  `responding_to.author` of their replies) are set to `null`, and their IDs are listed in
  `rehydrated/missing_comments.txt`. All messages are kept so that the annotations stay aligned.
- **Edited comments.** The API returns the current version of each comment. A comment that was
  edited after collection may differ from the text that was annotated.
- **Author names.** `AuthorName` is the author's current YouTube handle, which may have changed.
- `NumberOfLikes` and `Timestamp` are not updated: they keep their values at collection time.
- For any problem or additional inquiry you can contact the corresponding author at "bassidavide94@gmail.com"

Rehydrated data contains third-party content. Use it in line with the
[YouTube API Services Terms of Service](https://developers.google.com/youtube/terms/api-services-terms-of-service)
and do not redistribute it. The included `.gitignore` keeps the `rehydrated/` folder out of git.
