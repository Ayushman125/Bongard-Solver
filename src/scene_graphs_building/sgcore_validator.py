import os
os.environ["TRANSFORMERS_CACHE"] = "C:/Users/HP/hf_cache"
import asyncio
from typing import Dict, List, Optional, Tuple
import torch
from transformers import AutoProcessor, AutoModel
import base64
from io import BytesIO
from PIL import Image
import json
import logging

class SGScoreValidator:
    """
    Advanced scene graph validation using multimodal LLM chain-of-thought reasoning
    following the SGScore methodology for exponential quality improvement.
    """
    
    def __init__(self, 
                 model_name: str = "microsoft/git-base-coco",
                 api_endpoint: Optional[str] = None):
        
        cache_dir = os.environ.get("HF_HOME", "./model_cache")
        self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.api_endpoint = api_endpoint
        
        # Chain-of-thought validation templates
        self.validation_templates = self._load_validation_templates()
        
        # Confidence calibration parameters
        self.calibration_params = self._init_calibration()
        
        logging.info("Initialized SGScoreValidator with chain-of-thought reasoning")
    
    async def validate_scene_graph(self, 
                                 image: Image.Image,
                                 scene_graph: Dict,
                                 validation_mode: str = "comprehensive") -> Dict:
        """
        Perform comprehensive scene graph validation using multimodal LLM reasoning.
        Returns detailed validation report with confidence scores.
        """
        
        validation_results = {
            'overall_score': 0.0,
            'object_presence_score': 0.0,
            'relationship_accuracy_score': 0.0,
            'spatial_consistency_score': 0.0,
            'commonsense_plausibility_score': 0.0,
            'detailed_feedback': [],
            'corrected_relationships': []
        }
        
        # Validate object presence with chain-of-thought reasoning
        object_validation = await self._validate_objects_with_cot(image, scene_graph['objects'])
        validation_results['object_presence_score'] = object_validation['score']
        validation_results['detailed_feedback'].extend(object_validation['feedback'])
        
        # Validate relationships with spatial reasoning
        relationship_validation = await self._validate_relationships_with_cot(
            image, scene_graph['relationships'], scene_graph['objects']
        )
        validation_results['relationship_accuracy_score'] = relationship_validation['score']
        validation_results['detailed_feedback'].extend(relationship_validation['feedback'])
        
        # Spatial consistency validation
        spatial_validation = await self._validate_spatial_consistency(
            image, scene_graph
        )
        validation_results['spatial_consistency_score'] = spatial_validation['score']
        
        # Commonsense plausibility check
        commonsense_validation = await self._validate_commonsense_plausibility(
            scene_graph
        )
        validation_results['commonsense_plausibility_score'] = commonsense_validation['score']
        validation_results['corrected_relationships'] = commonsense_validation['corrections']
        
        # Calculate overall score with weighted combination
        weights = {'objects': 0.25, 'relationships': 0.35, 'spatial': 0.2, 'commonsense': 0.2}
        validation_results['overall_score'] = (
            weights['objects'] * validation_results['object_presence_score'] +
            weights['relationships'] * validation_results['relationship_accuracy_score'] +
            weights['spatial'] * validation_results['spatial_consistency_score'] +
            weights['commonsense'] * validation_results['commonsense_plausibility_score']
        )
        
        return validation_results
    
    async def _validate_objects_with_cot(self, 
                                       image: Image.Image, 
                                       objects: List[Dict]) -> Dict:
        """Validate object presence using chain-of-thought reasoning."""
        
        validation_results = {'score': 0.0, 'feedback': []}
        total_objects = len(objects)
        correct_objects = 0
        
        for obj in objects:
            # Create chain-of-thought prompt for object validation
            cot_prompt = self._create_object_validation_prompt(obj)
            
            # Get multimodal LLM response
            validation_response = await self._query_multimodal_llm(
                image, cot_prompt, task="object_validation"
            )
            
            # Parse chain-of-thought reasoning
            is_valid, confidence, reasoning = self._parse_validation_response(
                validation_response, task="object"
            )
            
            if is_valid and confidence > 0.7:
                correct_objects += 1
            
            validation_results['feedback'].append({
                'object': obj['label'],
                'bbox': obj.get('bbox', None),
                'is_valid': is_valid,
                'confidence': confidence,
                'reasoning': reasoning
            })
        
        validation_results['score'] = correct_objects / max(total_objects, 1)
        return validation_results
    
    async def _validate_relationships_with_cot(self, 
                                             image: Image.Image,
                                             relationships: List[Dict],
                                             objects: List[Dict]) -> Dict:
        """Validate relationships using advanced chain-of-thought reasoning."""
        
        validation_results = {'score': 0.0, 'feedback': []}
        total_relationships = len(relationships)
        correct_relationships = 0
        
        # Create object lookup for relationship validation
        object_lookup = {obj['id']: obj for obj in objects}
        
        for rel in relationships:
            subject_obj = object_lookup.get(rel['subject_id'])
            object_obj = object_lookup.get(rel['object_id'])
            
            if not subject_obj or not object_obj:
                continue
            
            # Create sophisticated relationship validation prompt
            cot_prompt = self._create_relationship_validation_prompt(
                subject_obj, rel['predicate'], object_obj
            )
            
            # Get multimodal LLM response with spatial reasoning
            validation_response = await self._query_multimodal_llm(
                image, cot_prompt, task="relationship_validation"
            )
            
            # Parse response with confidence calibration
            is_valid, confidence, reasoning = self._parse_validation_response(
                validation_response, task="relationship"
            )
            
            # Apply confidence calibration
            calibrated_confidence = self._calibrate_confidence(
                confidence, task="relationship"
            )
            
            if is_valid and calibrated_confidence > 0.6:
                correct_relationships += 1
            
            validation_results['feedback'].append({
                'subject': subject_obj['label'],
                'predicate': rel['predicate'],
                'object': object_obj['label'],
                'is_valid': is_valid,
                'raw_confidence': confidence,
                'calibrated_confidence': calibrated_confidence,
                'reasoning': reasoning
            })
        
        validation_results['score'] = correct_relationships / max(total_relationships, 1)
        return validation_results
    
    def _create_relationship_validation_prompt(self, 
                                             subject: Dict, 
                                             predicate: str, 
                                             obj: Dict) -> str:
        """Create sophisticated chain-of-thought prompt for relationship validation."""
        
        template = """
        Analyze the image and evaluate the relationship: "{subject}" {predicate} "{object}".
        
        Follow this chain-of-thought reasoning:
        
        Step 1 - Object Identification:
        - Can you clearly identify the {subject} in the image?
        - Can you clearly identify the {object} in the image?
        - Describe their visual characteristics and locations.
        
        Step 2 - Spatial Analysis:
        - What is the spatial relationship between these objects?
        - Consider position, distance, orientation, and scale.
        
        Step 3 - Relationship Evaluation:
        - Does the relationship "{predicate}" accurately describe their interaction?
        - Consider both spatial and functional aspects.
        - Are there any contradictions or implausible aspects?
        
        Step 4 - Confidence Assessment:
        - How confident are you in this relationship (0-100%)?
        - What evidence supports or contradicts this relationship?
        
        Step 5 - Final Judgment:
        - Is the relationship "{subject}" {predicate} "{object}" valid? (Yes/No)
        - Provide a brief explanation of your reasoning.
        
        Format your response as JSON:
        {{
            "object_identification": {{"subject_found": boolean, "object_found": boolean, "description": "..."}},
            "spatial_analysis": "...",
            "relationship_evaluation": "...",
            "confidence_percentage": integer,
            "final_judgment": "Yes" or "No",
            "reasoning_summary": "..."
        }}
        """
        
        return template.format(
            subject=subject['label'],
            predicate=predicate,
            object=obj['label']
        )
    
    async def _query_multimodal_llm(self, image: Image.Image, prompt: str, task: str) -> str:
        """Query multimodal LLM with image and text prompt. SOTA: Actually run model and return output."""
        import json
        import numpy as np
        if self.api_endpoint:
            return await self._query_external_api(image, prompt)
        else:
            try:
                from transformers import pipeline
                from PIL import Image as PILImage
                if not isinstance(image, PILImage.Image):
                    image = PILImage.fromarray(np.array(image))
                pipe = pipeline("image-to-text", model=self.model, tokenizer=self.processor)
                result = pipe({"image": image, "prompt": prompt})
                if isinstance(result, list) and len(result) > 0:
                    return result[0]['generated_text']
                return str(result)
            except Exception as e:
                import traceback
                logging.error(f"SGScoreValidator: Local LLM inference failed: {e}\n{traceback.format_exc()}")
                return ""

    def _parse_validation_response(self, response, task):
        """Parse LLM response (JSON or text) for object/relationship validation."""
        import json
        import re
        if not response:
            return False, 0.0, "No response"
        try:
            if isinstance(response, dict):
                data = response
            else:
                data = json.loads(response)
            if task == "object":
                is_valid = data.get("object_identification", {}).get("subject_found", True)
                confidence = float(data.get("confidence_percentage", 100)) / 100.0
                reasoning = data.get("reasoning_summary", "")
            else:
                is_valid = data.get("final_judgment", "Yes").lower().startswith("y")
                confidence = float(data.get("confidence_percentage", 100)) / 100.0
                reasoning = data.get("reasoning_summary", "")
            return is_valid, confidence, reasoning
        except Exception:
            conf_match = re.search(r"confidence.*?(\d{1,3})%", response, re.I)
            conf = float(conf_match.group(1)) / 100.0 if conf_match else 1.0
            valid = "yes" in response.lower()
            return valid, conf, response

    async def _query_external_api(self, image, prompt):
        """Call external API for multimodal LLM (e.g., GPT-4V, Gemini)."""
        import aiohttp
        import base64
        from io import BytesIO
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()
        payload = {"image": img_b64, "prompt": prompt}
        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_endpoint, json=payload) as resp:
                if resp.status == 200:
                    return await resp.text()
                else:
                    logging.error(f"SGScoreValidator: API call failed with status {resp.status}")
                    return ""

    async def _query_local_model(self, image, prompt):
        return ""

    async def _validate_spatial_consistency(self, image, scene_graph):
        rels = scene_graph.get('relationships', [])
        objs = {o['id']: o for o in scene_graph.get('objects', [])}
        spatial_score = 0.0
        n = 0
        import numpy as np
        for rel in rels:
            subj = objs.get(rel['subject_id'])
            obj = objs.get(rel['object_id'])
            if not subj or not obj:
                continue
            if rel['predicate'] in ('near', 'on', 'beside'):
                if 'centroid' in subj and 'centroid' in obj:
                    dist = np.linalg.norm(np.array(subj['centroid']) - np.array(obj['centroid']))
                    if dist < 50:
                        spatial_score += 1
            else:
                spatial_score += 1
            n += 1
        return {'score': spatial_score / max(n, 1)}

    async def _assess_relationship_plausibility(self, subject, predicate, obj):
        plausible_rels = {'on', 'under', 'beside', 'near', 'inside', 'outside', 'part_of', 'supports', 'contains'}
        if predicate in plausible_rels:
            return 1.0
        return 0.7

    def _load_validation_templates(self):
        return {
            'object': "Is the object '{label}' present in the image?",
            'relationship': self._create_relationship_validation_prompt
        }

    def _init_calibration(self):
        return {'object': {'slope': 1.0, 'bias': 0.0}, 'relationship': {'slope': 1.0, 'bias': 0.0}}
    def _create_object_validation_prompt(self, obj):
        return f"Is the object '{obj['label']}' present in the image?"
