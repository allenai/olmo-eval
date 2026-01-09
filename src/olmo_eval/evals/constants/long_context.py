# =============================================================================
# HELMET Benchmark Suites
# =============================================================================

HELMET_SUITES: dict[str, tuple[str, ...]] = {
    # Citation tasks
    "helmet_cite__131072::suite": (
        "helmet_alce_asqa_700__131072::std",
        "helmet_alce_qampari_700__131072::std",
    ),
    "helmet_cite__16384::suite": (
        "helmet_alce_asqa_75__16384::std",
        "helmet_alce_qampari_75__16384::std",
    ),
    "helmet_cite__32768::suite": (
        "helmet_alce_asqa_165__32768::std",
        "helmet_alce_qampari_165__32768::std",
    ),
    "helmet_cite__65536::suite": (
        "helmet_alce_asqa_345__65536::std",
        "helmet_alce_qampari_345__65536::std",
    ),
    "helmet_cite__8192::suite": (
        "helmet_alce_asqa_30__8192::std",
        "helmet_alce_qampari_30__8192::std",
    ),
    # In-context learning tasks
    "helmet_icl__131072::suite": (
        "helmet_icl_banking77_5900shot_balance__131072::std",
        "helmet_icl_clinic150_7050shot_balance__131072::std",
        "helmet_icl_nlu_8296shot_balance__131072::std",
        "helmet_icl_trec_coarse_6600shot_balance__131072::std",
        "helmet_icl_trec_fine_6400shot_balance__131072::std",
    ),
    "helmet_icl__16384::suite": (
        "helmet_icl_banking77_720shot_balance__16384::std",
        "helmet_icl_clinic150_880shot_balance__16384::std",
        "helmet_icl_nlu_1020shot_balance__16384::std",
        "helmet_icl_trec_coarse_800shot_balance__16384::std",
        "helmet_icl_trec_fine_800shot_balance__16384::std",
    ),
    "helmet_icl__32768::suite": (
        "helmet_icl_banking77_1450shot_balance__32768::std",
        "helmet_icl_clinic150_1750shot_balance__32768::std",
        "helmet_icl_nlu_2040shot_balance__32768::std",
        "helmet_icl_trec_coarse_1600shot_balance__32768::std",
        "helmet_icl_trec_fine_1600shot_balance__32768::std",
    ),
    "helmet_icl__65536::suite": (
        "helmet_icl_banking77_2900shot_balance__65536::std",
        "helmet_icl_clinic150_3525shot_balance__65536::std",
        "helmet_icl_nlu_4080shot_balance__65536::std",
        "helmet_icl_trec_coarse_3300shot_balance__65536::std",
        "helmet_icl_trec_fine_3200shot_balance__65536::std",
    ),
    "helmet_icl__8192::suite": (
        "helmet_icl_banking77_360shot_balance__8192::std",
        "helmet_icl_clinic150_440shot_balance__8192::std",
        "helmet_icl_nlu_510shot_balance__8192::std",
        "helmet_icl_trec_coarse_400shot_balance__8192::std",
        "helmet_icl_trec_fine_400shot_balance__8192::std",
    ),
    # Long QA tasks
    "helmet_longqa__131072::suite": (
        "helmet_infbench_choice_eng_130862__131072::std",
        "helmet_infbench_qa_eng_130862__131072::std",
        "helmet_narrativeqa_130772__131072::std",
    ),
    "helmet_longqa__16384::suite": (
        "helmet_infbench_choice_eng_16174__16384::std",
        "helmet_infbench_qa_eng_16174__16384::std",
        "helmet_narrativeqa_16084__16384::std",
    ),
    "helmet_longqa__32768::suite": (
        "helmet_infbench_choice_eng_32558__32768::std",
        "helmet_infbench_qa_eng_32558__32768::std",
        "helmet_narrativeqa_32468__32768::std",
    ),
    "helmet_longqa__65536::suite": (
        "helmet_infbench_choice_eng_65326__65536::std",
        "helmet_infbench_qa_eng_65326__65536::std",
        "helmet_narrativeqa_65236__65536::std",
    ),
    "helmet_longqa__8192::suite": (
        "helmet_infbench_choice_eng_7982__8192::std",
        "helmet_infbench_qa_eng_7982__8192::std",
        "helmet_narrativeqa_7892__8192::std",
    ),
    # NIAH (Needle in a Haystack) tasks
    "helmet_niah__131072::suite": (
        "helmet_ruler_cwe__131072::std",
        "helmet_ruler_fwe__131072::std",
        "helmet_ruler_niah_mk_1__131072::std",
        "helmet_ruler_niah_mk_2__131072::std",
        "helmet_ruler_niah_mk_3__131072::std",
        "helmet_ruler_niah_mq__131072::std",
        "helmet_ruler_niah_mv__131072::std",
        "helmet_ruler_niah_s_1__131072::std",
        "helmet_ruler_niah_s_2__131072::std",
        "helmet_ruler_niah_s_3__131072::std",
        "helmet_ruler_qa_1__131072::std",
        "helmet_ruler_qa_2__131072::std",
        "helmet_ruler_vt__131072::std",
    ),
    "helmet_niah__65536::suite": (
        "helmet_ruler_cwe__65536::std",
        "helmet_ruler_fwe__65536::std",
        "helmet_ruler_niah_mk_1__65536::std",
        "helmet_ruler_niah_mk_2__65536::std",
        "helmet_ruler_niah_mk_3__65536::std",
        "helmet_ruler_niah_mq__65536::std",
        "helmet_ruler_niah_mv__65536::std",
        "helmet_ruler_niah_s_1__65536::std",
        "helmet_ruler_niah_s_2__65536::std",
        "helmet_ruler_niah_s_3__65536::std",
        "helmet_ruler_qa_1__65536::std",
        "helmet_ruler_qa_2__65536::std",
        "helmet_ruler_vt__65536::std",
    ),
    # RAG tasks
    "helmet_rag__131072::suite": (
        "helmet_kilt_hotpotqa__131072::std",
        "helmet_kilt_nq__131072::std",
        "helmet_kilt_popqa_3__131072::std",
        "helmet_kilt_triviaqa__131072::std",
    ),
    "helmet_rag__16384::suite": (
        "helmet_kilt_hotpotqa__16384::std",
        "helmet_kilt_nq__16384::std",
        "helmet_kilt_popqa_3__16384::std",
        "helmet_kilt_triviaqa__16384::std",
    ),
    "helmet_rag__32768::suite": (
        "helmet_kilt_hotpotqa__32768::std",
        "helmet_kilt_nq__32768::std",
        "helmet_kilt_popqa_3__32768::std",
        "helmet_kilt_triviaqa__32768::std",
    ),
    "helmet_rag__65536::suite": (
        "helmet_kilt_hotpotqa__65536::std",
        "helmet_kilt_nq__65536::std",
        "helmet_kilt_popqa_3__65536::std",
        "helmet_kilt_triviaqa__65536::std",
    ),
    "helmet_rag__8192::suite": (
        "helmet_kilt_hotpotqa__8192::std",
        "helmet_kilt_nq__8192::std",
        "helmet_kilt_popqa_3__8192::std",
        "helmet_kilt_triviaqa__8192::std",
    ),
    # Recall tasks
    "helmet_recall__131072::suite": (
        "helmet_json_kv__131072::std",
        "helmet_recall_ruler_niah_mk_2__131072::std",
        "helmet_recall_ruler_niah_mk_3__131072::std",
        "helmet_recall_ruler_niah_mv__131072::std",
    ),
    "helmet_recall__16384::suite": (
        "helmet_json_kv__16384::std",
        "helmet_ruler_niah_mk_2__16384::std",
        "helmet_ruler_niah_mk_3__16384::std",
        "helmet_ruler_niah_mv__16384::std",
    ),
    "helmet_recall__32768::suite": (
        "helmet_json_kv__32768::std",
        "helmet_ruler_niah_mk_2__32768::std",
        "helmet_ruler_niah_mk_3__32768::std",
        "helmet_ruler_niah_mv__32768::std",
    ),
    "helmet_recall__65536::suite": (
        "helmet_json_kv__65536::std",
        "helmet_recall_ruler_niah_mk_2__65536::std",
        "helmet_recall_ruler_niah_mk_3__65536::std",
        "helmet_recall_ruler_niah_mv__65536::std",
    ),
    "helmet_recall__8192::suite": (
        "helmet_json_kv__8192::std",
        "helmet_ruler_niah_mk_2__8192::std",
        "helmet_ruler_niah_mk_3__8192::std",
        "helmet_ruler_niah_mv__8192::std",
    ),
    # Reranking tasks
    "helmet_rerank__131072::suite": ("helmet_msmarco_rerank_psg__131072::std",),
    "helmet_rerank__16384::suite": ("helmet_msmarco_rerank_psg__16384::std",),
    "helmet_rerank__32768::suite": ("helmet_msmarco_rerank_psg__32768::std",),
    "helmet_rerank__65536::suite": ("helmet_msmarco_rerank_psg__65536::std",),
    "helmet_rerank__8192::suite": ("helmet_msmarco_rerank_psg__8192::std",),
    # Summarization tasks
    "helmet_summ__131072::suite": (
        "helmet_infbench_sum_eng_129672__131072::std",
        "helmet_multi_lexsum_130372__131072::std",
    ),
    "helmet_summ__16384::suite": (
        "helmet_infbench_sum_eng_14984__16384::std",
        "helmet_multi_lexsum_15684__16384::std",
    ),
    "helmet_summ__32768::suite": (
        "helmet_infbench_sum_eng_31368__32768::std",
        "helmet_multi_lexsum_32068__32768::std",
    ),
    "helmet_summ__65536::suite": (
        "helmet_infbench_sum_eng_64136__65536::std",
        "helmet_multi_lexsum_64836__65536::std",
    ),
    "helmet_summ__8192::suite": (
        "helmet_infbench_sum_eng_6792__8192::std",
        "helmet_multi_lexsum_7492__8192::std",
    ),
}
"""HELMET long-context benchmark suites organized by task type and context length."""


# =============================================================================
# RULER Benchmark Suites
# =============================================================================

RULER_SUITES: dict[str, tuple[str, ...]] = {
    # 4096 context length
    "ruler_niah__4096::suite": (
        "ruler_niah_s_1__4096::std",
        "ruler_niah_s_2__4096::std",
        "ruler_niah_s_3__4096::std",
        "ruler_niah_mk_1__4096::std",
        "ruler_niah_mk_2__4096::std",
        "ruler_niah_mk_3__4096::std",
        "ruler_niah_mv__4096::std",
        "ruler_niah_mq__4096::std",
    ),
    "ruler_multi_hop_tracing__4096::suite": ("ruler_vt__4096::std",),
    "ruler_aggregation__4096::suite": (
        "ruler_cwe__4096::std",
        "ruler_fwe__4096::std",
    ),
    "ruler_qa__4096::suite": (
        "ruler_qa_1__4096::std",
        "ruler_qa_2__4096::std",
    ),
    # 8192 context length
    "ruler_niah__8192::suite": (
        "ruler_niah_s_1__8192::std",
        "ruler_niah_s_2__8192::std",
        "ruler_niah_s_3__8192::std",
        "ruler_niah_mk_1__8192::std",
        "ruler_niah_mk_2__8192::std",
        "ruler_niah_mk_3__8192::std",
        "ruler_niah_mv__8192::std",
        "ruler_niah_mq__8192::std",
    ),
    "ruler_multi_hop_tracing__8192::suite": ("ruler_vt__8192::std",),
    "ruler_aggregation__8192::suite": (
        "ruler_cwe__8192::std",
        "ruler_fwe__8192::std",
    ),
    "ruler_qa__8192::suite": (
        "ruler_qa_1__8192::std",
        "ruler_qa_2__8192::std",
    ),
    # 16384 context length
    "ruler_niah__16384::suite": (
        "ruler_niah_s_1__16384::std",
        "ruler_niah_s_2__16384::std",
        "ruler_niah_s_3__16384::std",
        "ruler_niah_mk_1__16384::std",
        "ruler_niah_mk_2__16384::std",
        "ruler_niah_mk_3__16384::std",
        "ruler_niah_mv__16384::std",
        "ruler_niah_mq__16384::std",
    ),
    "ruler_multi_hop_tracing__16384::suite": ("ruler_vt__16384::std",),
    "ruler_aggregation__16384::suite": (
        "ruler_cwe__16384::std",
        "ruler_fwe__16384::std",
    ),
    "ruler_qa__16384::suite": (
        "ruler_qa_1__16384::std",
        "ruler_qa_2__16384::std",
    ),
    # 32768 context length
    "ruler_niah__32768::suite": (
        "ruler_niah_s_1__32768::std",
        "ruler_niah_s_2__32768::std",
        "ruler_niah_s_3__32768::std",
        "ruler_niah_mk_1__32768::std",
        "ruler_niah_mk_2__32768::std",
        "ruler_niah_mk_3__32768::std",
        "ruler_niah_mv__32768::std",
        "ruler_niah_mq__32768::std",
    ),
    "ruler_multi_hop_tracing__32768::suite": ("ruler_vt__32768::std",),
    "ruler_aggregation__32768::suite": (
        "ruler_cwe__32768::std",
        "ruler_fwe__32768::std",
    ),
    "ruler_qa__32768::suite": (
        "ruler_qa_1__32768::std",
        "ruler_qa_2__32768::std",
    ),
    # 65536 context length
    "ruler_niah__65536::suite": (
        "ruler_niah_s_1__65536::std",
        "ruler_niah_s_2__65536::std",
        "ruler_niah_s_3__65536::std",
        "ruler_niah_mk_1__65536::std",
        "ruler_niah_mk_2__65536::std",
        "ruler_niah_mk_3__65536::std",
        "ruler_niah_mv__65536::std",
        "ruler_niah_mq__65536::std",
    ),
    "ruler_multi_hop_tracing__65536::suite": ("ruler_vt__65536::std",),
    "ruler_aggregation__65536::suite": (
        "ruler_cwe__65536::std",
        "ruler_fwe__65536::std",
    ),
    "ruler_qa__65536::suite": (
        "ruler_qa_1__65536::std",
        "ruler_qa_2__65536::std",
    ),
    # 131072 context length
    "ruler_niah__131072::suite": (
        "ruler_niah_s_1__131072::std",
        "ruler_niah_s_2__131072::std",
        "ruler_niah_s_3__131072::std",
        "ruler_niah_mk_1__131072::std",
        "ruler_niah_mk_2__131072::std",
        "ruler_niah_mk_3__131072::std",
        "ruler_niah_mv__131072::std",
        "ruler_niah_mq__131072::std",
    ),
    "ruler_multi_hop_tracing__131072::suite": ("ruler_vt__131072::std",),
    "ruler_aggregation__131072::suite": (
        "ruler_cwe__131072::std",
        "ruler_fwe__131072::std",
    ),
    "ruler_qa__131072::suite": (
        "ruler_qa_1__131072::std",
        "ruler_qa_2__131072::std",
    ),
}
"""RULER long-context benchmark suites organized by task type and context length."""
