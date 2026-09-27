# Third-party materials and source terms

No new blanket software or data license is assigned by this packaging step. This is an initial research-review snapshot; included dependencies and third-party materials retain their own terms.

- The fine-tuned checkpoint derives from GPT-2. The upstream MIT notice is included in `licenses/gpt2-LICENSE.txt`; see [OpenAI GPT-2](https://github.com/openai/gpt-2/blob/master/LICENSE) and the [public model card](https://huggingface.co/openai-community/gpt2).
- Wikipedia passages were obtained through [`wikimedia/wikipedia`, 20231101.en](https://huggingface.co/datasets/wikimedia/wikipedia). Its dataset card lists CC-BY-SA-3.0 and GFDL and links the Wikimedia terms. These passages are not public-domain merely because they are publicly accessible.
- arXiv passages were obtained through [`ccdv/arxiv-summarization`](https://huggingface.co/datasets/ccdv/arxiv-summarization). The available dataset card does not provide a clear blanket license. Public accessibility does not itself establish unrestricted redistribution rights; source-paper terms apply.

The frozen corpus retained source-category labels but not per-document URLs/IDs. Precise document attribution and redistribution scope for those passages remain a public-release limitation to resolve. The exact corpus is preserved in this initial private source snapshot for reproducibility. The synthetic target registry and the mixed background corpus are distinct materials.

Names in upstream license notices identify third-party copyright holders, not the submission's authors. All reference links in this notice concern public dependencies/datasets, not the authors' repository.
