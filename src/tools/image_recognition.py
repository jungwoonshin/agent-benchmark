"""Image recognition tool for processing images from PDFs and browser navigation."""

import logging
from typing import Any, Dict, List, Optional, Union

from ..models import Attachment


class ImageRecognition:
    """Tool for recognizing and analyzing images from PDFs and browser navigation."""

    def __init__(self, logger: logging.Logger, llm_service=None):
        """
        Initialize ImageRecognition tool.

        Args:
            logger: Logger instance for logging.
            llm_service: Optional LLM service for visual processing.
        """
        self.logger = logger
        self.llm_service = llm_service

    def set_llm_service(self, llm_service):
        """Set the LLM service for visual processing."""
        self.llm_service = llm_service

    def recognize_images_from_pdf(
        self,
        attachment: Attachment,
        options: Optional[Dict[str, Any]] = None,
        problem: Optional[str] = None,
        query_analysis: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Dict[str, Any]]:
        """
        Extract and recognize images from a PDF file with smart content filtering.

        Args:
            attachment: PDF attachment to process.
            options: Optional dict with 'page_range' key for specific pages.
            problem: Optional problem description for relevance filtering.
            query_analysis: Optional query analysis for relevance filtering.

        Returns:
            If problem/query_analysis provided: Dictionary with structured data:
            {
                'type': 'pdf',
                'filename': str,
                'sections': [{'title': str, 'page': int, 'content': str}, ...],
                'image_analysis': str,
                'full_text': str
            }
            Otherwise: Combined text and image analysis results as string (backward compatibility).
        """
        if options is None:
            options = {}

        try:
            import fitz  # type: ignore # PyMuPDF
        except ImportError:
            error_msg = 'PyMuPDF not available. Install with: uv pip install PyMuPDF'
            self.logger.warning(error_msg)
            return f'Error: {error_msg}'

        self.logger.info(
            f'Extracting and recognizing images from PDF: {attachment.filename}'
        )

        try:
            # Open PDF from bytes
            pdf_document = fitz.open(stream=attachment.data, filetype='pdf')

            # Determine page range
            page_range = options.get('page_range', None)
            if page_range and isinstance(page_range, list) and len(page_range) == 2:
                start_page = max(0, page_range[0] - 1)  # Convert to 0-based
                end_page = min(len(pdf_document), page_range[1])  # End is inclusive
                pages = range(start_page, end_page)
                self.logger.info(f'Reading PDF pages {page_range[0]}-{page_range[1]}')
            else:
                pages = range(len(pdf_document))
                self.logger.info(f'Reading all {len(pdf_document)} pages of PDF')

            # Step 1: Extract structure (headers/titles) first if problem/query_analysis provided
            relevant_sections = None
            pdf_structure = None
            if problem and query_analysis and self.llm_service:
                self.logger.info(
                    'Extracting PDF structure (headers/titles) for relevance filtering...'
                )
                pdf_structure = self._extract_pdf_structure(pdf_document, pages)
                if pdf_structure:
                    self.logger.info(
                        f'Found {len(pdf_structure)} sections in PDF structure'
                    )
                    # Filter sections by relevance
                    relevant_sections = self._filter_relevant_sections(
                        pdf_structure, problem, query_analysis
                    )
                    if relevant_sections:
                        self.logger.info(
                            f'Identified {len(relevant_sections)} relevant sections out of {len(pdf_structure)} total'
                        )
                    else:
                        self.logger.warning(
                            'No relevant sections found - will skip content extraction (only extract images)'
                        )

            # Step 2: Extract text and images from specified pages
            # Only extract full content from relevant sections if filtering was applied
            text_parts = []
            extracted_images = []
            structured_sections = []  # For structured return format

            for page_num in pages:
                page = pdf_document[page_num]

                # Check if this page should be included (if relevance filtering is active)
                should_extract_text = True
                if relevant_sections is not None:  # None means filtering was attempted
                    # If filtering was attempted but no relevant sections found, skip all text
                    if len(relevant_sections) == 0:
                        should_extract_text = False
                        self.logger.debug(
                            f'Skipping page {page_num + 1} - no relevant sections found in PDF'
                        )
                    else:
                        # Check if any relevant section is on this page
                        page_sections = [
                            s
                            for s in relevant_sections
                            if s.get('page', 0) == page_num + 1
                        ]
                        should_extract_text = len(page_sections) > 0
                        if not should_extract_text:
                            self.logger.debug(
                                f'Skipping page {page_num + 1} - no relevant sections on this page'
                            )

                if should_extract_text:
                    # Extract text from page
                    page_text = page.get_text()
                    if page_text.strip():
                        # If relevance filtering was applied (relevant_sections is not None)
                        if relevant_sections is not None:
                            # Extract only from relevant sections
                            page_sections = [
                                s
                                for s in relevant_sections
                                if s.get('page', 0) == page_num + 1
                            ]
                            if page_sections:
                                # Extract text for relevant sections on this page using section_index
                                section_texts = []
                                # Get page blocks for block-based extraction
                                blocks = page.get_text('dict')['blocks']
                                blocks_text_list = []
                                for block in blocks:
                                    block_text = ''
                                    if 'lines' in block:
                                        for line in block['lines']:
                                            if 'spans' in line:
                                                for span in line['spans']:
                                                    block_text += span.get('text', '')
                                    blocks_text_list.append(block_text)

                                for section in page_sections:
                                    section_index = section.get('section_index')
                                    section_title = section.get('title', '')

                                    if section_index is not None and pdf_structure:
                                        # Find the section in pdf_structure by section_index
                                        current_section = None
                                        for s in pdf_structure:
                                            if s.get('section_index') == section_index:
                                                current_section = s
                                                break

                                        if current_section:
                                            # Get block index for current section
                                            start_block_idx = current_section.get(
                                                'block_index', 0
                                            )

                                            # Find next section's block index (section_index + 1)
                                            next_block_idx = len(blocks)
                                            next_section_index = section_index + 1
                                            for s in pdf_structure:
                                                if (
                                                    s.get('section_index')
                                                    == next_section_index
                                                    and s.get('page') == page_num + 1
                                                ):
                                                    next_block_idx = s.get(
                                                        'block_index', len(blocks)
                                                    )
                                                    break

                                            # Extract text from blocks between start and next
                                            section_blocks_text = blocks_text_list[
                                                start_block_idx:next_block_idx
                                            ]
                                            section_text = '\n'.join(
                                                section_blocks_text
                                            ).strip()

                                            if section_text:
                                                section_texts.append(
                                                    f'[Section: {section_title}]\n{section_text}'
                                                )
                                                # Add to structured sections
                                                structured_sections.append(
                                                    {
                                                        'title': section_title,
                                                        'page': page_num + 1,
                                                        'content': section_text,
                                                        'section_index': section_index,
                                                    }
                                                )
                                            else:
                                                # Fallback: try string matching if block extraction yielded no text
                                                self.logger.debug(
                                                    f'Block-based extraction failed for section {section_index}, trying string matching'
                                                )
                                                if section_title in page_text:
                                                    start_idx = page_text.find(
                                                        section_title
                                                    )
                                                    if start_idx >= 0:
                                                        # Find next section on same page
                                                        remaining_text = page_text[
                                                            start_idx:
                                                        ]
                                                        next_section_idx = len(
                                                            remaining_text
                                                        )
                                                        for s in pdf_structure:
                                                            if (
                                                                s.get('page')
                                                                == page_num + 1
                                                                and s.get(
                                                                    'section_index', -1
                                                                )
                                                                > section_index
                                                            ):
                                                                other_title = s.get(
                                                                    'title', ''
                                                                )
                                                                if (
                                                                    other_title
                                                                    in remaining_text
                                                                ):
                                                                    other_idx = remaining_text.find(
                                                                        other_title
                                                                    )
                                                                    if (
                                                                        other_idx > 0
                                                                        and other_idx
                                                                        < next_section_idx
                                                                    ):
                                                                        next_section_idx = other_idx
                                                        section_text = remaining_text[
                                                            :next_section_idx
                                                        ].strip()
                                                        if section_text:
                                                            section_texts.append(
                                                                f'[Section: {section_title}]\n{section_text}'
                                                            )
                                                            # Add to structured sections
                                                            structured_sections.append(
                                                                {
                                                                    'title': section_title,
                                                                    'page': page_num
                                                                    + 1,
                                                                    'content': section_text,
                                                                    'section_index': section_index,
                                                                }
                                                            )

                                if section_texts:
                                    text_parts.append(
                                        f'[Page {page_num + 1}]\n'
                                        + '\n\n'.join(section_texts)
                                    )
                                else:
                                    # Skip if section extraction failed (no fallback to full page text)
                                    self.logger.debug(
                                        f'Section extraction failed for page {page_num + 1}, skipping text extraction'
                                    )
                            # else: No relevant sections on this page, skip text extraction
                        else:
                            # No relevance filtering applied (problem/query_analysis not provided)
                            # Extract full page text for backward compatibility
                            text_parts.append(f'[Page {page_num + 1}]\n{page_text}')

                # Always extract images from all pages (images might be relevant even if text isn't)
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    try:
                        # Get image data
                        xref = img[0]
                        base_image = pdf_document.extract_image(xref)
                        image_bytes = base_image['image']
                        image_ext = base_image['ext']

                        # Store image with metadata
                        extracted_images.append(
                            {
                                'data': image_bytes,
                                'page': page_num + 1,
                                'index': img_index,
                                'ext': image_ext,
                            }
                        )
                        self.logger.debug(
                            f'Extracted image {img_index + 1} from page {page_num + 1} '
                            f'({len(image_bytes)} bytes, format: {image_ext})'
                        )
                    except Exception as e:
                        self.logger.warning(
                            f'Failed to extract image {img_index} from page {page_num + 1}: {e}'
                        )

            pdf_document.close()

            # Combine text parts
            combined_text = (
                '\n\n'.join(text_parts) if text_parts else '[No text content found]'
            )

            # Build context that includes relevant section titles if available
            context_for_images = combined_text
            if relevant_sections is not None and len(relevant_sections) > 0:
                relevant_titles_info = (
                    '\n\nRelevant Sections Identified:\n'
                    + '\n'.join(
                        f'- Page {s.get("page", "?")}: {s.get("title", "")}'
                        for s in relevant_sections
                    )
                )
                context_for_images = (
                    relevant_titles_info + '\n\n' + combined_text
                    if combined_text != '[No text content found]'
                    else relevant_titles_info
                )
                self.logger.info(
                    f'Including {len(relevant_sections)} relevant section titles in image analysis context'
                )

            # Process images with visual LLM if available
            image_analysis = ''
            if extracted_images and self.llm_service:
                self.logger.info(
                    f'Found {len(extracted_images)} image(s) in PDF. Processing with visual LLM...'
                )
                image_analysis = self._process_images_with_visual_llm(
                    extracted_images,
                    task_description=problem
                    or 'Analyze the images from this PDF and extract key information.',
                    context_text=context_for_images,
                    source_type='PDF',
                    source_name=attachment.filename,
                )
            elif extracted_images:
                # If no visual LLM available, just note that images were found
                image_info = ', '.join(
                    f'page {img["page"]} (image {img["index"] + 1})'
                    for img in extracted_images
                )
                image_analysis = (
                    f'\n\n[Note: {len(extracted_images)} image(s) found in PDF ({image_info}), '
                    f'but visual LLM not available for analysis]'
                )

            # Combine results
            result = combined_text
            if image_analysis:
                result += '\n\n'
                result += 'IMAGE ANALYSIS (from visual LLM):\n'
                result += '\n'
                result += image_analysis

            self.logger.info(
                f'Successfully processed PDF {attachment.filename}. '
                f'Text length: {len(combined_text)}, Images: {len(extracted_images)}'
            )

            # Return structured data if problem/query_analysis provided and we have sections
            if problem and query_analysis and structured_sections:
                return {
                    'type': 'pdf',
                    'filename': attachment.filename,
                    'sections': structured_sections,
                    'image_analysis': image_analysis,
                    'full_text': result,  # Include full combined text as fallback
                }

            # Backward compatibility: return string
            return result

        except Exception as e:
            self.logger.error(
                f'Failed to process PDF {attachment.filename}: {e}', exc_info=True
            )
            return f'Error: Failed to process PDF {attachment.filename}: {str(e)}'

    def recognize_images_from_browser(
        self,
        screenshot_data: bytes,
        context: Optional[Dict[str, Any]] = None,
        task_description: Optional[str] = None,
    ) -> str:
        """
        Recognize and analyze images from browser navigation (screenshots).
        Args:
            screenshot_data: Screenshot image data as bytes.
            context: Optional context dictionary with page information.
            task_description: Optional task description for analysis.
        Returns:
            Analysis result from visual LLM.
        """
        if not screenshot_data:
            return 'Error: No screenshot data provided'

        if not self.llm_service:
            return '[Visual LLM not available for screenshot analysis]'

        self.logger.info(
            f'Processing browser screenshot with visual LLM ({len(screenshot_data)} bytes)'
        )

        context_str = ''
        if context:
            context_items = []
            for key, value in context.items():
                if key in ('screenshot', 'image') or 'image' in key.lower():
                    continue
                context_items.append(f'- {key}: {str(value)[:500]}')
            if context_items:
                context_str = '\n'.join(context_items)

        final_task_description = (
            task_description
            or 'Analyze this webpage screenshot and determine the next action.'
        )

        return self._process_images_with_visual_llm(
            images=[screenshot_data],
            task_description=final_task_description,
            context_text=context_str,
            source_type='browser',
            source_name=context.get('url', 'unknown URL') if context else 'unknown URL',
        )

    def recognize_images(
        self,
        images: List[bytes],
        task_description: str,
        context: Optional[Dict[str, Any]] = None,
        source_type: str = 'general',
        source_name: Optional[str] = None,
    ) -> str:
        """
        Recognize and analyze a list of images using visual LLM.

        Args:
            images: List of image data as bytes.
            task_description: Description of what to analyze in the images.
            context: Optional context dictionary.
            source_type: Type of source (e.g., 'PDF', 'browser', 'general').
            source_name: Optional name of the source.

        Returns:
            Analysis result from visual LLM.
        """
        if not images:
            return '[No images provided]'

        if not self.llm_service:
            return '[Visual LLM not available]'

        self.logger.info(
            f'Processing {len(images)} image(s) from {source_type} with visual LLM'
        )

        return self._process_images_with_visual_llm(
            images,
            task_description=task_description,
            context_text=str(context) if context else '',
            source_type=source_type,
            source_name=source_name or 'unknown',
        )

    def _process_images_with_visual_llm(
        self,
        images: List[Any],
        task_description: str,
        context_text: str = '',
        source_type: str = 'general',
        source_name: str = 'unknown',
    ) -> str:
        """
        Process images with visual LLM.
        Args:
            images: List of images (can be dicts with 'data' key or bytes).
            task_description: Description of what to analyze in the images.
            context_text: Text context for analysis.
            source_type: Type of source.
            source_name: Name of the source.
        Returns:
            Analysis result.
        """
        if not self.llm_service:
            return '[Visual LLM not available]'

        try:
            # Build content list with text and images
            content_items = []

            # System prompt is now universal and direct
            system_prompt = """You are an expert at analyzing images to extract relevant information.
Focus on answering the user's request based on the visual content.
Be concise and directly address the user's query.
If the image does not contain relevant information, state that clearly.
Do not describe the image unless asked to."""

            # User prompt is now a clear instruction
            text_prompt = f"""Task: {task_description}
Source: {source_type} ({source_name})"""

            if context_text:
                text_prompt += f'\n\nRelevant Text Context:\n{context_text[:2000]}'  # Limit context length

            text_prompt += '\n\nPlease analyze the following image(s) and provide a direct answer to the task.'

            content_items.append({'type': 'text', 'text': text_prompt})

            # Add all images
            for i, img in enumerate(images):
                try:
                    # Handle both dict format (from PDF) and bytes format
                    if isinstance(img, dict):
                        image_data = img['data']
                        image_format = img.get('ext', 'auto')
                    else:
                        image_data = img
                        image_format = 'auto'

                    image_content = self.llm_service.create_image_content(
                        image_data, image_format=image_format
                    )
                    content_items.append(image_content)
                    self.logger.debug(f'Added image {i + 1} to visual LLM request')
                except Exception as e:
                    self.logger.warning(f'Failed to encode image {i + 1}: {e}')

            if len(content_items) == 1:  # Only text, no valid images added
                return '[Failed to process images with visual LLM - no valid images]'

            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': content_items},
            ]

            self.logger.info(
                f'Processing {len(images)} image(s) from {source_type} with visual LLM...'
            )

            # Use visual LLM to analyze images
            analysis = self.llm_service.call_with_images(
                messages=messages,
                temperature=0.1,  # Lower temperature for more focused analysis
                max_tokens=8192,  # Increased from 4000 to reduce truncation risk
            )

            self.logger.info(f'Visual LLM analysis completed: {len(analysis)} chars')
            return analysis

        except Exception as e:
            self.logger.error(
                f'Failed to process images with visual LLM: {e}', exc_info=True
            )
            return f'[Error processing images with visual LLM: {str(e)}]'

    def _extract_pdf_structure(
        self, pdf_document: Any, pages: range
    ) -> List[Dict[str, Any]]:
        """
        Extract headers and titles from PDF structure.

        Args:
            pdf_document: PyMuPDF document object.
            pages: Range of pages to process.

        Returns:
            List of sections with title, page, and level information.
        """
        sections = []
        try:
            # Try to extract structure using PyMuPDF's structure
            for page_num in pages:
                page = pdf_document[page_num]
                # Get page text blocks
                blocks = page.get_text('dict')['blocks']

                # Extract potential headers (larger text, bold, at start of blocks)
                for block_idx, block in enumerate(blocks):
                    if 'lines' in block:
                        for line in block['lines']:
                            if 'spans' in line:
                                for span in line['spans']:
                                    text = span.get('text', '').strip()
                                    if not text:
                                        continue

                                    font_size = span.get('size', 0)
                                    flags = span.get('flags', 0)
                                    is_bold = flags & 16  # Bit 4 indicates bold

                                    # Heuristic: Headers are typically:
                                    # - Larger font size (>= 12pt)
                                    # - Bold
                                    # - Short text (typically < 200 chars)
                                    # - At start of line
                                    # - May be all caps or title case
                                    if (
                                        font_size >= 12
                                        and is_bold
                                        and len(text) < 200
                                        and len(text.split()) <= 15
                                    ):
                                        sections.append(
                                            {
                                                'title': text,
                                                'page': page_num + 1,
                                                'level': 1 if font_size >= 16 else 2,
                                                'font_size': font_size,
                                                'block_index': block_idx,
                                            }
                                        )

            # Also try to extract from table of contents if available
            toc = pdf_document.get_toc()
            if toc:
                for item in toc:
                    level, title, page_num_toc = item
                    if page_num_toc - 1 in pages:  # Convert to 0-based
                        # Try to find matching block index on the target page
                        toc_page = pdf_document[page_num_toc - 1]
                        toc_blocks = toc_page.get_text('dict')['blocks']
                        block_index = None
                        # Search for the title in blocks to get approximate position
                        for block_idx, block in enumerate(toc_blocks):
                            # Extract text from this block
                            block_text = ''
                            if 'lines' in block:
                                for line in block['lines']:
                                    if 'spans' in line:
                                        for span in line['spans']:
                                            block_text += span.get('text', '')
                            if (
                                title.lower() in block_text.lower()[:200]
                            ):  # Check first 200 chars
                                block_index = block_idx
                                break
                        sections.append(
                            {
                                'title': title,
                                'page': page_num_toc,
                                'level': level,
                                'font_size': 14,  # Default for TOC entries
                                'block_index': block_index
                                if block_index is not None
                                else 0,
                            }
                        )

            # Remove duplicates (same title on same page)
            seen = set()
            unique_sections = []
            for section in sections:
                key = (section['title'], section['page'])
                if key not in seen:
                    seen.add(key)
                    unique_sections.append(section)

            # Sort sections by page number, then by block_index within page
            unique_sections.sort(key=lambda s: (s['page'], s.get('block_index', 0)))

            # Assign 0-based sequential section_index to each section
            for idx, section in enumerate(unique_sections):
                section['section_index'] = idx

            self.logger.debug(
                f'Extracted {len(unique_sections)} unique sections from PDF structure with indices 0-{len(unique_sections) - 1}'
            )
            return unique_sections

        except Exception as e:
            self.logger.warning(f'Failed to extract PDF structure: {e}')
            return []

    def _filter_relevant_sections(
        self,
        sections: List[Dict[str, Any]],
        problem: str,
        query_analysis: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Filter PDF sections by relevance to problem and query analysis using LLM.

        Args:
            sections: List of section dictionaries with title, page, level.
            problem: Problem description.
            query_analysis: Query analysis dictionary.

        Returns:
            List of relevant sections.
        """
        if not self.llm_service or not sections:
            return sections

        try:
            # Build prompt for relevance checking
            sections_text = '\n'.join(
                [
                    f'- Page {s["page"]}, Level {s["level"]}: {s["title"]}'
                    for s in sections
                ]
            )

            explicit_requirements = query_analysis.get('explicit_requirements', [])
            implicit_requirements = query_analysis.get('implicit_requirements', [])

            system_prompt = """You are an expert at analyzing document structure and determining relevance.
Your task is to identify which sections of a PDF document are relevant to answering a specific question.

Consider:
1. The explicit requirements mentioned in the question
2. The implicit requirements inferred from the question
3. The context and domain of the question
4. Whether the section title suggests it contains relevant information

Return a JSON object with a "relevant_titles" key containing an array of relevant section titles."""

            user_prompt = f"""Problem/Question: {problem}

Explicit Requirements: {', '.join(explicit_requirements) if explicit_requirements else 'None specified'}
Implicit Requirements: {', '.join(implicit_requirements) if implicit_requirements else 'None specified'}

PDF Section Titles:
{sections_text}

Identify which section titles are relevant to answering the problem/question.
Return a JSON object with a "relevant_titles" key containing an array of relevant section titles (as strings).
Example: {{"relevant_titles": ["Introduction", "Methodology", "Results"]}}"""

            response = self.llm_service.call_with_system_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
                response_format={'type': 'json_object'},
            )

            # Parse response
            import json

            from ..utils.json_utils import extract_json_from_text

            json_text = extract_json_from_text(response)
            response_data = json.loads(json_text)

            # Extract relevant titles (handle different response formats)
            relevant_titles = set()
            if isinstance(response_data, dict):
                # Look for common keys
                if 'relevant_titles' in response_data:
                    relevant_titles = set(response_data['relevant_titles'])
                elif 'relevant_sections' in response_data:
                    relevant_titles = set(response_data['relevant_sections'])
                elif 'titles' in response_data:
                    relevant_titles = set(response_data['titles'])
                elif 'sections' in response_data:
                    relevant_titles = set(response_data['sections'])
                else:
                    # Try to find any array value
                    for value in response_data.values():
                        if isinstance(value, list):
                            relevant_titles = set(value)
                            break
            elif isinstance(response_data, list):
                relevant_titles = set(response_data)

            # Normalize titles (trim whitespace, case-insensitive matching)
            relevant_titles_normalized = {t.strip().lower() for t in relevant_titles}

            # Filter sections by relevant titles (case-insensitive matching)
            # Preserve section_index from original sections
            relevant_sections = [
                s
                for s in sections
                if s['title'].strip().lower() in relevant_titles_normalized
            ]

            self.logger.info(
                f'LLM identified {len(relevant_titles)} relevant titles out of {len(sections)} total sections'
            )
            if relevant_sections:
                indices = [s.get('section_index', '?') for s in relevant_sections]
                self.logger.debug(f'Relevant section indices: {indices}')

            # Return only relevant sections (no fallback to all sections)
            return relevant_sections

        except Exception as e:
            self.logger.warning(
                f'Failed to filter sections by relevance: {e}. Will skip content extraction.'
            )
            return []  # Return empty list if filtering fails - no fallback to all sections
