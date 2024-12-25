import gradio as gr
from dataclasses import dataclass
from typing import List, Optional, Literal
from PIL import Image
import torch
import random
from . import elements
from . import constants
from argparse import ArgumentParser


ITEM_PER_PAGE = 12


class OutfitManager:
    
    def __init__(self):
        self.items = []
        
    def image_preview(self):
        images = (
            [item.image for item in self.items]
            if self.items else None
        )
        
        return images

    def add(
        self, 
        image, 
        description,
        category = None
    ) -> None:
        self.items.append(
            elements.Item(
                image=image, 
                description=description,
                category=category,
            )
        )
        
        return self.image_preview()

    def delete(
        self, 
        idx = int,
    ):
        if isinstance(idx, int) and (0 <= idx < len(self.items)):
            self.items.pop(idx)

        return self.image_preview()

    def clear(self):
        self.items = []
        
        return self.image_preview()
    

class DBBuffer:
    
    def __init__(self):
        self.items = []


def demo(
    model,
    item_loader,
    task: Literal['cp', 'cir'] = 'cp',
    indexer = None
):
    if task == 'cir' and indexer is None:
        raise ValueError(
            "Indexer must be provided for type='cir'"
        )

    manager = OutfitManager()
    db_candidates_buffer = DBBuffer()
    
    @torch.no_grad()
    def cp():
        return float(model.predict([elements.Outfit(items=manager.items)])[0])
    
    @torch.no_grad()
    def cir(query):
        query = elements.Query(
            query=query,
            items=manager.items
        )

        embeddings = model.embed_query([query]).detach().cpu().numpy()

        res = indexer.search_items(
            embeddings=embeddings,
            top_k=12
        )[0]
        images = [
            item_loader(int(i)).image 
            for i in res
        ]
        
        return images
    
    with gr.Blocks() as demo:
        outfit_selected_idx = gr.State(value=None)
        
        gr.Markdown(
            "## Compatibility Prediction"
        )
        
        gr.Markdown(
            "### 1. Add Items"
        )
        with gr.Row(equal_height=True):
            with gr.Column(scale=2, variant='panel'):
                input_category = gr.Dropdown(
                    label="Category",
                    choices=constants.CATEGORIES,
                    value=None,
                )
                input_description = gr.Textbox(
                    label="Enter Description",
                )
                input_image = gr.Image(
                    label="Upload Image",
                    type="pil",
                )
                
            with gr.Column(scale=8, variant='panel'):
                db_candidates_gallery = gr.Gallery(
                    allow_preview=False,
                    show_label=True,
                    columns=4,
                    rows=3,
                    type="pil",
                )
                db_candidates_gallery_page = gr.Dropdown(
                    choices=[1], value=1,
                    label="Page",
                )

            def update_db_candidates_gallery_category(selected_category):
                db_candidates_buffer.items=item_loader.paginate_items(page=1, item_per_page=ITEM_PER_PAGE, category=selected_category)
                total_pages = item_loader.total_pages(item_per_page=ITEM_PER_PAGE, category=selected_category)
                return (
                    gr.update(value=[item.image for item in db_candidates_buffer.items]),
                    gr.update(choices=[i for i in range(1, total_pages + 1)], value=1) # Reset Value
                )
                
            input_category.change(
                update_db_candidates_gallery_category,
                inputs=[input_category],
                outputs=[db_candidates_gallery, db_candidates_gallery_page],
            )
            
            def update_db_candidates_gallery_by_page(selected_page, selected_category):
                db_candidates_buffer.items=item_loader.paginate_items(page=selected_page, item_per_page=ITEM_PER_PAGE, category=selected_category)
                return gr.update(value=[item.image for item in db_candidates_buffer.items])
            
            db_candidates_gallery_page.change(
                update_db_candidates_gallery_by_page,
                inputs=[db_candidates_gallery_page, input_category],
                outputs=[db_candidates_gallery],
            )
            
            def update_item_preview(evt: gr.SelectData):
                selected_item = db_candidates_buffer.items[evt.index]
                return (
                    gr.update(value=selected_item.image),
                    gr.update(value=selected_item.description)
                )
                
            db_candidates_gallery.select(
                update_item_preview,
                inputs=None,
                outputs=[input_image, input_description],
            )
        with gr.Row():
            btn_add_to_inputs = gr.Button("Add Item to Outfit")
            
        gr.Markdown(
            "### 2. Check Outfit"
        )
        with gr.Row(equal_height=True):
            with gr.Column(scale=8, variant='panel'):
                inputs_gallery = gr.Gallery(
                    allow_preview=False,
                    show_label=True,
                    columns=6,
                    rows=1,
                )
                with gr.Row():
                    btn_inputs_gallery_clear = gr.Button(
                        "Clear All Items",
                    )
                    btn_inputs_gallery_delete = gr.Button(
                        "Delete Item",
                    )
                    
        btn_add_to_inputs.click(
            lambda img, desc, cat: (
                manager.add(img, desc, cat), gr.update(value=None)
            ),
            inputs=[input_image, input_description, input_category], 
            outputs=[inputs_gallery, outfit_selected_idx]
        )
        def get_selected_idx(selected: gr.SelectData):
            return selected.index
        inputs_gallery.select(
            get_selected_idx,
            inputs=None,
            outputs=outfit_selected_idx
        )
        btn_inputs_gallery_clear.click(
            lambda: (
                manager.clear(), gr.update(value=None)
            ),
            inputs=None,
            outputs=[inputs_gallery, outfit_selected_idx]
        )
        btn_inputs_gallery_delete.click(
            lambda idx: (
                manager.delete(idx), gr.update(value=None)
            ),
            inputs=outfit_selected_idx, 
            outputs=[inputs_gallery, outfit_selected_idx]
        )
        
        if task == 'cp':   
            gr.Markdown(
                "### 3. Compute Score"
            )
            with gr.Row(equal_height=True):
                with gr.Column(scale=2, variant='panel'):
                    btn_evaluate = gr.Button(
                        "Evaluate",
                        variant="primary"
                    )
                with gr.Column(scale=8, variant='panel'):
                    score = gr.Textbox(
                        label="Compatibility Score",
                        interactive=False
                    )
            btn_evaluate.click(
                cp,
                inputs=None,
                outputs=score
            )
                    
        elif task == 'cir':
            gr.Markdown(
                "### 3. Search"
            )
            with gr.Row(equal_height=True):
                with gr.Column(scale=2, variant='panel'):
                    query = gr.Radio(
                        label="Category",
                        choices=constants.CATEGORIES,
                        value=None,
                    )
                    # query = gr.Textbox(
                    #     label="Enter Query",
                    # )
                    btn_search = gr.Button(
                        "Search",
                        variant="primary"
                    )
                with gr.Column(scale=8, variant='panel'):
                    search_result_gallery = gr.Gallery(
                        allow_preview=False,
                        show_label=True,
                        columns=4,
                        rows=3,
                        type="pil",
                    )
            btn_search.click(
                cir,
                inputs=query,
                outputs=search_result_gallery
            )

    demo.launch()
