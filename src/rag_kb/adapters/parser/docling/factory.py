"""Frozen Docling 2.114.0 converter construction."""

from __future__ import annotations

from pathlib import Path

from docling.backend.md_backend import MarkdownBackendOptions
from docling.datamodel.accelerator_options import (
    AcceleratorDevice,
    AcceleratorOptions,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    ConvertPipelineOptions,
    PdfPipelineOptions,
    RapidOcrOptions,
)
from docling.document_converter import (
    CsvFormatOption,
    DocumentConverter,
    ExcelFormatOption,
    HTMLFormatOption,
    MarkdownFormatOption,
    PdfFormatOption,
    PowerpointFormatOption,
    WordFormatOption,
)

from rag_kb.domain import ParserLimits, ParsingPreset


ALLOWED_FORMATS = (
    InputFormat.MD,
    InputFormat.PDF,
    InputFormat.DOCX,
    InputFormat.HTML,
    InputFormat.CSV,
    InputFormat.PPTX,
    InputFormat.XLSX,
)


def build_docling_converter(
    preset: ParsingPreset,
    *,
    artifacts_path: Path,
    limits: ParserLimits,
) -> DocumentConverter:
    """Build one local-only converter for a frozen parsing preset."""

    multimodal = preset is ParsingPreset.MULTIMODAL_LOCAL_V2
    accelerator_options = AcceleratorOptions(
        num_threads=1,
        device=AcceleratorDevice.CPU,
    )
    simple_options = ConvertPipelineOptions(
        document_timeout=limits.document_timeout_seconds,
        accelerator_options=accelerator_options,
        enable_remote_services=False,
        allow_external_plugins=False,
        artifacts_path=artifacts_path,
        do_picture_classification=False,
        do_picture_description=False,
        do_chart_extraction=False,
    )
    pdf_options = PdfPipelineOptions(
        document_timeout=limits.document_timeout_seconds,
        accelerator_options=accelerator_options,
        enable_remote_services=False,
        allow_external_plugins=False,
        artifacts_path=artifacts_path,
        do_picture_classification=False,
        do_picture_description=False,
        do_chart_extraction=False,
        generate_page_images=multimodal,
        generate_picture_images=multimodal,
        generate_table_images=False,
        do_table_structure=True,
        do_ocr=True,
        do_code_enrichment=False,
        do_formula_enrichment=False,
        ocr_options=RapidOcrOptions(
            # RapidOCR's Chinese recognition model is bilingual and includes
            # Latin/English glyphs. Supplying both model-set names makes
            # RapidOCR silently select the first set, so freeze the bilingual
            # Chinese set explicitly.
            lang=["chinese"],
            backend="onnxruntime",
            force_full_page_ocr=False,
        ),
        ocr_batch_size=1,
        layout_batch_size=1,
        table_batch_size=1,
    )
    return DocumentConverter(
        allowed_formats=list(ALLOWED_FORMATS),
        format_options={
            InputFormat.MD: MarkdownFormatOption(
                pipeline_options=simple_options,
                backend_options=(
                    MarkdownBackendOptions(
                        fetch_images=True,
                        enable_local_fetch=True,
                        enable_remote_fetch=False,
                        max_image_data_base64_bytes=limits.max_total_asset_bytes,
                    )
                    if preset is ParsingPreset.MULTIMODAL_LOCAL_V2
                    else MarkdownBackendOptions()
                ),
            ),
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
            InputFormat.DOCX: WordFormatOption(
                pipeline_options=simple_options,
            ),
            InputFormat.HTML: HTMLFormatOption(
                pipeline_options=simple_options,
            ),
            InputFormat.CSV: CsvFormatOption(
                pipeline_options=simple_options,
            ),
            InputFormat.PPTX: PowerpointFormatOption(
                pipeline_options=simple_options,
            ),
            InputFormat.XLSX: ExcelFormatOption(
                pipeline_options=simple_options,
            ),
        },
    )
