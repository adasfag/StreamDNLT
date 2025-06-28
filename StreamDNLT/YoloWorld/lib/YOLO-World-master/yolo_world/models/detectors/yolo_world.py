
from typing import List, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
from mmdet.structures import OptSampleList, SampleList
from mmyolo.models.detectors import YOLODetector
from mmyolo.registry import MODELS
from lib.groundingdino.models.GroundingDINO.transformer_vanilla import TransformerDecoderSimple
from lib.groundingdino.models.GroundingDINO.transformer_vanilla import CosineAttentionFusion
import copy




@MODELS.register_module()
class YOLOWorldDetector(YOLODetector):
    """Implementation of YOLOW Series"""

    def __init__(self,
                 *args,
                 mm_neck: bool = False,
                 num_train_classes=80,
                 num_test_classes=80,
                 **kwargs) -> None:
        self.mm_neck = mm_neck
        self.num_train_classes = num_train_classes
        self.num_test_classes = num_test_classes
        super().__init__(*args, **kwargs)
        
        self.num_level=3
        
        
        self.num_query_memory=6
        self.query_memory_tgt_adapter_1= nn.Embedding(self.num_query_memory, 256)
        self.query_memory_tgt_pos_adapter_1 = nn.Embedding(self.num_query_memory, 256)
        
        self.appearance_feature_fusion_adapter_1=TransformerDecoderSimple(num_layers=1,d_model=256,nhead=4)
        self.text_feature_fusion_adapter_1=TransformerDecoderSimple(num_layers=1,d_model=256,nhead=4)
        self.text_project_1=nn.Linear(in_features=768,out_features=256)
        





        self.update_text_project=nn.Linear(in_features=768,out_features=512)
        self.inv_update_text_project=nn.Linear(in_features=512,out_features=768)

        self.temporal_feature_fusion_adapter_1=TransformerDecoderSimple(num_layers=1,d_model=256,nhead=4)
        

        self.temporal_update_text_adapter=TransformerDecoderSimple(num_layers=1,d_model=512,nhead=4)
        

        self.memory_sequence_adapter_1=CosineAttentionFusion(256)
        
        self.memory_sequence_update_text_adapter=CosineAttentionFusion(512)
        
        
        
        self.memory_query_init_1=nn.Parameter(torch.randn(6, 256))
        
        self.combined_visual_feature_text1=nn.Embedding(16, 512)
        self.combined_visual_feature_text2=nn.Embedding(16, 512)
        self.combined_visual_feature_text3=nn.Embedding(16, 512)

        self.combined_visual_feature_pos1=nn.Embedding(16, 512)
        self.combined_visual_feature_pos2=nn.Embedding(16, 512)
        self.combined_visual_feature_pos3=nn.Embedding(16, 512)

        self.combined_visual_feature_fusion_adapter_1=TransformerDecoderSimple(num_layers=6,d_model=512,nhead=4)
        self.combined_visual_feature_fusion_adapter_2=TransformerDecoderSimple(num_layers=6,d_model=512,nhead=4)
        self.combined_visual_feature_fusion_adapter_3=TransformerDecoderSimple(num_layers=6,d_model=512,nhead=4)
        
        self.combined_visual_feature_fusion_adapter=nn.ModuleList([
            self.combined_visual_feature_fusion_adapter_1,
            self.combined_visual_feature_fusion_adapter_2,
            self.combined_visual_feature_fusion_adapter_3
            
        ])

        self.combined_visual_feature_text=nn.ParameterList([
            self.combined_visual_feature_text1,
            self.combined_visual_feature_text2,
             self.combined_visual_feature_text3
            
        ])

        self.combined_visual_feature_text_pos=nn.ParameterList([
            self.combined_visual_feature_pos1,
            self.combined_visual_feature_pos2,
            self.combined_visual_feature_pos3
            
        ])

        self.visual_feature_project=nn.Linear(in_features=256,out_features=512)
        

        
        
        
        self.query_memory_tgt_adapter_2= nn.Embedding(self.num_query_memory, 512)
        self.query_memory_tgt_pos_adapter_2 = nn.Embedding(self.num_query_memory, 512)
        
        self.appearance_feature_fusion_adapter_2=TransformerDecoderSimple(num_layers=1,d_model=512,nhead=4)
        self.text_feature_fusion_adapter_2=TransformerDecoderSimple(num_layers=1,d_model=512,nhead=4)
        self.text_project_2=nn.Linear(in_features=768,out_features=512)
        
        self.temporal_feature_fusion_adapter_2=TransformerDecoderSimple(num_layers=1,d_model=512,nhead=4)
        
        self.memory_sequence_adapter_2=CosineAttentionFusion(512)
        
        self.memory_query_init_2=nn.Parameter(torch.randn(6, 512))
        
        
        self.query_memory_tgt_adapter_3= nn.Embedding(self.num_query_memory, 512)
        self.query_memory_tgt_pos_adapter_3 = nn.Embedding(self.num_query_memory, 512)
        
        self.appearance_feature_fusion_adapter_3=TransformerDecoderSimple(num_layers=1,d_model=512,nhead=4)
        self.text_feature_fusion_adapter_3=TransformerDecoderSimple(num_layers=1,d_model=512,nhead=4)
        self.text_project_3=nn.Linear(in_features=768,out_features=512)
        
        self.temporal_feature_fusion_adapter_3=TransformerDecoderSimple(num_layers=1,d_model=512,nhead=4)
        
        self.memory_sequence_adapter_3=CosineAttentionFusion(512)
        
        self.memory_query_init_3=nn.Parameter(torch.randn(6, 512))
        
        
        
        
        self.query_memory_tgt_adapter= nn.ModuleList([
            self.query_memory_tgt_adapter_1,
            self.query_memory_tgt_adapter_2,
            self.query_memory_tgt_adapter_3
            
        ])
        self.query_memory_tgt_pos_adapter = nn.ModuleList([
             self.query_memory_tgt_pos_adapter_1,
             self.query_memory_tgt_pos_adapter_2,
             self.query_memory_tgt_pos_adapter_3
            
            
        ])
        
        self.appearance_feature_fusion_adapter=nn.ModuleList([
            self.appearance_feature_fusion_adapter_1,
            self.appearance_feature_fusion_adapter_2,
            self.appearance_feature_fusion_adapter_3
            
        ])
        self.text_feature_fusion_adapter=nn.ModuleList([
             self.text_feature_fusion_adapter_1,
              self.text_feature_fusion_adapter_2,
               self.text_feature_fusion_adapter_3
            
        ])
        
        self.text_project=nn.ModuleList([
            self.text_project_1,
            self.text_project_2,
            self.text_project_3
            
        ])
        
        self.temporal_feature_fusion_adapter=nn.ModuleList([
            self.temporal_feature_fusion_adapter_1,
            self.temporal_feature_fusion_adapter_2,
            self.temporal_feature_fusion_adapter_3
            
        ])
        
        self.memory_sequence_adapter=nn.ModuleList([
            self.memory_sequence_adapter_1,
            self.memory_sequence_adapter_2,
            self.memory_sequence_adapter_3
            
        ])
        
        
        self.memory_query_init=nn.ParameterList([
            self.memory_query_init_1,
            self.memory_query_init_2,
            self.memory_query_init_3
            
        ])
        
        
        self.pre_query=None
        self.pre_update_text_query=None
        
        
        
        
        
        
        
        
        
        
        
        
        

        
        
        
        
        
        
        
        self.query_spatial_tgt_adapter=copy.deepcopy(self.query_memory_tgt_adapter)
        self.query_spatial_tgt_pos_adapter = copy.deepcopy( self.query_memory_tgt_pos_adapter )
        self.appearance_spatial_feature_fusion_adapter=copy.deepcopy(self.appearance_feature_fusion_adapter)
        self.spatial_sequence_adapter_adapter=copy.deepcopy(self.memory_sequence_adapter)
        

        
        
        
        
        
        
        
        
        
        
         
        
        
        
        
        
        
        
        


    def loss_bak(self, batch_inputs,
             batch_data_samples) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        (batch_data_samples,bboxes_labels,img_metas,template_inputs)=batch_data_samples
        self.bbox_head.num_classes = self.num_train_classes
        



        
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                batch_data_samples,
                                                template_inputs)
        
        
        memory_img_feats=[]
        for index_i,img_feat in enumerate(img_feats):
            
            bs,c,h,w=img_feat.shape
        
            tgt_ = img_feat.permute(0,2,3,1).reshape(bs,h*w,c)
            memory_text_=txt_feats 

            
            query_=self.combine_feature_memory(tgt_,memory_text_,index_i)
            
            batches,num_query,num_channel=tgt_.shape
            times=4
            bs=batches//times

            query_=query_.reshape(times,bs,self.num_query_memory,num_channel)
            
            updated_query_=torch.zeros_like(query_)
            
            
            memory_query_init=self.memory_query_init[index_i].unsqueeze(0).repeat(bs, 1, 1)
            
            for index in range(times):
                if index<1:
                    pre_query=  memory_query_init
                    updated_query_[index]=pre_query
                else:
                    now_query_=query_[index-1]
                    pre_query=self.temporal_feature_fusion_adapter[index_i](tgt=pre_query,memory=now_query_,memory_pos=None)
                    updated_query_[index]=pre_query
            query_=query_.reshape(times*bs,self.num_query_memory,num_channel)
            updated_query_=updated_query_.reshape(times*bs,self.num_query_memory,num_channel)
            
            memory=self.memory_sequence_adapter[index_i](q=tgt_,k=updated_query_,v=updated_query_)
            memory_img_feats.append(memory.reshape(batches,h,w,c).permute(0,3,1,2))
        
        
        
        

        
        
        
        
        
        losses = self.bbox_head.loss(memory_img_feats, txt_feats, batch_data_samples,bboxes_labels,img_metas)
        return losses
    
    def loss(self, batch_inputs,
             batch_data_samples) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        (batch_data_samples,bboxes_labels,img_metas,template_inputs)=batch_data_samples
        self.bbox_head.num_classes = self.num_train_classes
        



        
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                batch_data_samples,
                                                template_inputs)
        
        
        memory_img_feats=[]
        for index_i,img_feat in enumerate(img_feats):
            
            bs,c,h,w=img_feat.shape
        
            tgt_ = img_feat.permute(0,2,3,1).reshape(bs,h*w,c)
            memory_text_=txt_feats 

            
            query_=self.combine_feature_memory(tgt_,memory_text_,index_i)
            
            batches,num_query,num_channel=tgt_.shape
            times=4
            bs=batches//times

            query_=query_.reshape(times,bs,self.num_query_memory,num_channel)
            
            updated_query_=torch.zeros_like(query_)
            
            
            memory_query_init=self.memory_query_init[index_i].unsqueeze(0).repeat(bs, 1, 1)
            
            for index in range(times):
                if index<1:
                    pre_query=  memory_query_init
                    updated_query_[index]=pre_query
                else:
                    now_query_=query_[index-1]
                    pre_query=self.temporal_feature_fusion_adapter[index_i](tgt=pre_query,memory=now_query_,memory_pos=None)
                    updated_query_[index]=pre_query
            query_=query_.reshape(times*bs,self.num_query_memory,num_channel)
            updated_query_=updated_query_.reshape(times*bs,self.num_query_memory,num_channel)
            
            memory=self.memory_sequence_adapter[index_i](q=tgt_,k=updated_query_,v=updated_query_)
            memory_img_feats.append(memory.reshape(batches,h,w,c).permute(0,3,1,2))
        





        
        updated_query_=[]
        for index_i,img_feat in enumerate(img_feats):
            
            bs,c,h,w=img_feat.shape
        
            tgt_ = img_feat.permute(0,2,3,1).reshape(bs,h*w,c)
    
            if index_i ==0:
                tgt_=self.visual_feature_project(tgt_)
            
            query_=self.combine_feature_memory_for_text(tgt_,index_i)
            updated_query_.append(query_)

        updated_query_=torch.cat(updated_query_,dim=1)
        memory_text_=self.update_text_project(txt_feats)
        updated_texts=self.memory_sequence_update_text_adapter(q=memory_text_,k=updated_query_,v=updated_query_)
        updated_texts=self.inv_update_text_project(updated_texts)

        txt_feats=updated_texts

        
        
        
        

        
        
        
        
        
        losses = self.bbox_head.loss(memory_img_feats, txt_feats, batch_data_samples,bboxes_labels,img_metas)
        return losses


    
    












    def combine_feature(self, tgt_,index):
        
        
        
        
        bs,num_query,num_channel=tgt_.shape
        
        query_ = (
                    self.query_spatial_tgt_adapter[index].weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
                ) 
        
        query_pos_ = (
                    self.query_spatial_tgt_pos_adapter[index].weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
                ) 
                
        query_=self.appearance_spatial_feature_fusion_adapter[index](tgt=query_,pos=query_pos_,memory=tgt_,memory_pos=None)
        
        
        
        return query_ 
    
    
    
    
    
    def combine_feature_memory(self, tgt_,memory_text_,index):
        
        
        
        
        bs,num_query,num_channel=tgt_.shape
        
        memory_text_=self.text_project[index](memory_text_)
        
        query_ = (
                    self.query_memory_tgt_adapter[index].weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
                ) 
        
        query_pos_ = (
                    self.query_memory_tgt_pos_adapter[index].weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
                ) 
                
        query_=self.appearance_feature_fusion_adapter[index](tgt=query_,pos=query_pos_,memory=tgt_,memory_pos=None)
         
        query_=self.text_feature_fusion_adapter[index](tgt=query_,pos=query_pos_,memory=memory_text_,memory_pos=None)
        
        
        
        return query_ 
    

    def combine_feature_memory_for_text(self, tgt_,index):
        
        
        
        
        bs,num_query,num_channel=tgt_.shape
        
        query_ = (
                    self.combined_visual_feature_text[index].weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
                ) 
        
        query_pos_ = (
                    self.combined_visual_feature_text_pos[index].weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
                ) 
                
        
     
        query_=self.combined_visual_feature_fusion_adapter[index](tgt=query_,pos=query_pos_,memory=tgt_,memory_pos=None)
         
      
        
        
        return query_ 
    
    
    
    
    
    


    

    
    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """
        (batch_data_samples,template)=batch_data_samples
        
        
        
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples,
                                                 template)
        
        
        memory_img_feats=[]
        memory_pre_query=[]
        
    
        for index_i,img_feat in enumerate(img_feats):
            
            
            
            bs,c,h,w=img_feat.shape
        
            tgt_ = img_feat.permute(0,2,3,1).reshape(bs,h*w,c)
            memory_text_=txt_feats 

            
            query_=self.combine_feature_memory(tgt_,memory_text_,index_i)
            
            
            
            
            
            if self.pre_query is None:
                pre_query=self.memory_query_init[index_i].unsqueeze(0)
                update_query=self.temporal_feature_fusion_adapter[index_i](tgt=pre_query,memory=query_,memory_pos=None)
            else:
                pre_query=self.pre_query[index_i]
                update_query=self.temporal_feature_fusion_adapter[index_i](tgt=pre_query,memory=query_,memory_pos=None)

            memory_pre_query.append(update_query)
            
            memory=self.memory_sequence_adapter[index_i](q=tgt_,k=pre_query,v=pre_query)
            memory_img_feats.append(memory.reshape(bs,h,w,c).permute(0,3,1,2))
            
        self.pre_query=memory_pre_query
        
        
        
        
        
        
        
        
        
        
        self.bbox_head.num_classes = txt_feats[0].shape[0]
        results_list = self.bbox_head.predict(memory_img_feats,
                                              txt_feats,
                                              batch_data_samples,
                                              rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        batch_data_samples[0].current_img_feature=img_feats
        batch_data_samples[0].current_txt_feats=txt_feats
        batch_data_samples[0].pre_query= self.pre_query
        return batch_data_samples
    
    def predict_update_text(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """
        (batch_data_samples,template)=batch_data_samples
        
        
        
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples,
                                                 template)
        
        if template is not None:
            memory_img_feats=[]
            memory_pre_query=[]
            
        
            for index_i,img_feat in enumerate(img_feats):
                
                
                
                bs,c,h,w=img_feat.shape
            
                tgt_ = img_feat.permute(0,2,3,1).reshape(bs,h*w,c)
                memory_text_=txt_feats 

                
                query_=self.combine_feature_memory(tgt_,memory_text_,index_i)
                
                
                
                
                
                if self.pre_query is None:
                    pre_query=self.memory_query_init[index_i].unsqueeze(0)
                    update_query=self.temporal_feature_fusion_adapter[index_i](tgt=pre_query,memory=query_,memory_pos=None)
                else:
                    pre_query=self.pre_query[index_i]
                    update_query=self.temporal_feature_fusion_adapter[index_i](tgt=pre_query,memory=query_,memory_pos=None)

                memory_pre_query.append(update_query)
                
                memory=self.memory_sequence_adapter[index_i](q=tgt_,k=pre_query,v=pre_query)
                memory_img_feats.append(memory.reshape(bs,h,w,c).permute(0,3,1,2))
                
            self.pre_query=memory_pre_query



            
            updated_query_=[]
            for index_i,img_feat in enumerate(img_feats):
                
                bs,c,h,w=img_feat.shape
            
                tgt_ = img_feat.permute(0,2,3,1).reshape(bs,h*w,c)
        
                if index_i ==0:
                    tgt_=self.visual_feature_project(tgt_)
                
                query_=self.combine_feature_memory_for_text(tgt_,index_i)
                updated_query_.append(query_)

            updated_query_=torch.cat(updated_query_,dim=1)
            memory_text_=self.update_text_project(txt_feats)
            updated_texts=self.memory_sequence_update_text_adapter(q=memory_text_,k=updated_query_,v=updated_query_)
            updated_texts=self.inv_update_text_project(updated_texts)

            txt_feats=updated_texts
        else:
            memory_img_feats=img_feats
            
            
            
        
        
        
        
        
        
        
        self.bbox_head.num_classes = txt_feats[0].shape[0]
        results_list = self.bbox_head.predict(memory_img_feats,
                                              txt_feats,
                                              batch_data_samples,
                                              rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        batch_data_samples[0].current_img_feature=img_feats
        batch_data_samples[0].current_txt_feats=txt_feats
        batch_data_samples[0].pre_query= self.pre_query
        return batch_data_samples

    def reparameterize(self, texts: List[List[str]]) -> None:
        
        self.texts = texts
        self.text_feats = self.backbone.forward_text(texts)

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        
    
        results = self.bbox_head.forward(img_feats, txt_feats)
        return results

    def extract_feat(
            self, batch_inputs,
            batch_data_samples,
            template_inputs=None) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        txt_feats = None
        if batch_data_samples is None:
            texts = self.texts
            txt_feats = self.text_feats
        elif isinstance(batch_data_samples,
                        dict) and 'texts' in batch_data_samples:
            texts = batch_data_samples['texts']
        elif isinstance(batch_data_samples, list) and hasattr(
                batch_data_samples[0], 'texts'):
            texts = [data_sample.texts for data_sample in batch_data_samples]
        elif hasattr(self, 'text_feats'):
            texts = self.texts
            txt_feats = self.text_feats
        else:
            raise TypeError('batch_data_samples should be dict or list.')
        if txt_feats is not None:
            
            img_feats = self.backbone.forward_image(batch_inputs)
        else:
            img_feats, txt_feats = self.backbone(batch_inputs, texts)
        
        
        
        
        
        
        
        if template_inputs is not None:
            template_img_feats = self.backbone.forward_image(template_inputs)
                
            enhance_img_feats=[]
            for index_i,img_feat in enumerate(img_feats):
                
                bs,c,h,w=img_feat.shape
            
                tgt_ = img_feat.permute(0,2,3,1).reshape(bs,h*w,c)
                
                
                template_img_feat=template_img_feats[index_i].detach()
                
                template_query_=template_img_feat.permute(0,2,3,1).reshape(bs,-1,c)

                
                query_=self.combine_feature(template_query_,index_i)

                
                enhance_img_feat=self.spatial_sequence_adapter_adapter[index_i](q=tgt_,k=query_,v=query_)
                enhance_img_feats.append(enhance_img_feat.reshape(bs,h,w,c).permute(0,3,1,2))
            img_feats=enhance_img_feats
        
        
            
            
            
            
            
        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)
        return img_feats, txt_feats


@MODELS.register_module()
class SimpleYOLOWorldDetector(YOLODetector):
    """Implementation of YOLO World Series"""

    def __init__(self,
                 *args,
                 mm_neck: bool = False,
                 num_train_classes=80,
                 num_test_classes=80,
                 prompt_dim=512,
                 num_prompts=80,
                 embedding_path='',
                 reparameterized=False,
                 freeze_prompt=False,
                 use_mlp_adapter=False,
                 **kwargs) -> None:
        self.mm_neck = mm_neck
        self.num_training_classes = num_train_classes
        self.num_test_classes = num_test_classes
        self.prompt_dim = prompt_dim
        self.num_prompts = num_prompts
        self.reparameterized = reparameterized
        self.freeze_prompt = freeze_prompt
        self.use_mlp_adapter = use_mlp_adapter
        super().__init__(*args, **kwargs)

        if not self.reparameterized:
            if len(embedding_path) > 0:
                import numpy as np
                self.embeddings = torch.nn.Parameter(
                    torch.from_numpy(np.load(embedding_path)).float())
            else:
                
                embeddings = nn.functional.normalize(torch.randn(
                    (num_prompts, prompt_dim)),
                                                     dim=-1)
                self.embeddings = nn.Parameter(embeddings)

            if self.freeze_prompt:
                self.embeddings.requires_grad = False
            else:
                self.embeddings.requires_grad = True

            if use_mlp_adapter:
                self.adapter = nn.Sequential(
                    nn.Linear(prompt_dim, prompt_dim * 2), nn.ReLU(True),
                    nn.Linear(prompt_dim * 2, prompt_dim))
            else:
                self.adapter = None

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        self.bbox_head.num_classes = self.num_training_classes
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        if self.reparameterized:
            losses = self.bbox_head.loss(img_feats, batch_data_samples)
        else:
            losses = self.bbox_head.loss(img_feats, txt_feats,
                                         batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """

        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)

        self.bbox_head.num_classes = self.num_test_classes
        if self.reparameterized:
            results_list = self.bbox_head.predict(img_feats,
                                                  batch_data_samples,
                                                  rescale=rescale)
        else:
            results_list = self.bbox_head.predict(img_feats,
                                                  txt_feats,
                                                  batch_data_samples,
                                                  rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        if self.reparameterized:
            results = self.bbox_head.forward(img_feats)
        else:
            results = self.bbox_head.forward(img_feats, txt_feats)
        return results

    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        
        img_feats, _ = self.backbone(batch_inputs, None)

        if not self.reparameterized:
            
            txt_feats = self.embeddings[None]
            if self.adapter is not None:
                txt_feats = self.adapter(txt_feats) + txt_feats
                txt_feats = nn.functional.normalize(txt_feats, dim=-1, p=2)
            txt_feats = txt_feats.repeat(img_feats[0].shape[0], 1, 1)
        else:
            txt_feats = None
        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)
        return img_feats, txt_feats
