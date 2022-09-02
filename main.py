from torch import autocast
import torch
from image_to_image import StableDiffusionImg2ImgPipeline, preprocess
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
	"CompVis/stable-diffusion-v1-4", 
    revision="fp16",
    torch_dtype=torch.float16,
	use_auth_token=True
).to("cuda")

def generate_original_image(characters,base_font):
    out = []
    for i in range(len(characters)):
        target = characters[i]
        img = Image.new('RGB', (512, 512),(255,255,255))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(base_font,size=192)
        bounds = []
        x,y,w,h = draw.textbbox((0,0),target,font=font)
        for count in range(4):
            dx = 30-x if count%2==0 else 512-30-w
            dy = 30-y if count<=1 else 512-30-h
            draw.text((dx,dy),target,(0,0,0),font=font)
            bounds.append((
                30+w if count%2==0 else 512-30-w+x,
                30+h if count<=1 else 512-30-h+y
            ))
        cv2_img = np.asarray(img)
        gray = cv2.cvtColor(cv2_img,cv2.COLOR_BGR2GRAY)
        _,th = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
        nlabels, _ = cv2.connectedComponents(cv2.bitwise_not(th))
        out.append([img,target,nlabels-1,bounds,h])
    return out


def extract_letters(img,bounds,blocks):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray[gray[:,:] > 64] = 255
    _,mono = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    mono = cv2.morphologyEx(mono, cv2.MORPH_CLOSE, np.ones((2,2),np.uint8))
    mono = cv2.morphologyEx(mono, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    contours,_ = cv2.findContours(cv2.bitwise_not(mono), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
        rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
        topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
        bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
        if (leftmost[0] < img.shape[0]/2 <= rightmost[0]) or (topmost[1] < img.shape[1]/2 <= bottommost[1]):
            mono = cv2.drawContours(mono,[cnt],0,255,-1)
    cv2.rectangle(mono,(bounds[0][0]+(bounds[1][0]-bounds[0][0])//4,0),(bounds[0][0]+3*(bounds[1][0]-bounds[0][0])//4,img.shape[1]),255,-1)
    cv2.rectangle(mono,(0,bounds[0][1]+(bounds[2][1]-bounds[0][1])//4),(img.shape[0],bounds[0][1]+3*(bounds[2][1]-bounds[0][1])//4),255,-1)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(cv2.bitwise_not(mono))
    sorted_nlabels = list(sorted(range(1,nlabels),key=lambda x:stats[x][4],reverse=True))
    out = np.ones(img.shape,np.uint8)*255
    i = 0
    last_stat = 0
    for nlabel in sorted_nlabels:
        if blocks > i or stats[nlabel][4] > stats[sorted_nlabels[1]][4]*0.1:
            out[labels==nlabel] = [0,0,0]
        else:
            break
        i+= 1
    return out

def trim_letters(img):
    parts,r = [np.hsplit(vimg,2) for vimg in np.vsplit(img,2)]
    parts.extend(r)
    out = []
    for p in parts:
        xmost = np.where(np.any(p==0,axis=1))
        ymost = np.where(np.any(p==0,axis=0))
        xo = xmost[0] if len(xmost[0])>=1 else [0,p.shape[0]]
        yo = ymost[0] if len(ymost[0])>=1 else [0,p.shape[1]]
        out.append(p[xo[0]:xo[-1],yo[0]:yo[-1]])
    return out



def match_letters(parts):
    width_list = [p.shape[0] for p in parts]
    height_list = [p.shape[0] for p in parts]
    maximum_image = np.argmax(width_list+height_list)
    if (max(width_list)-min(width_list))+(max(height_list)-min(height_list))>30:
        return False
    else:
        maximum_hist = cv2.calcHist([parts[maximum_image]],[0],None,[256],[0,256])
        for i,p in enumerate(parts):
            if i != maximum_image:
                hist = cv2.calcHist([p],[0],None,[256],[0,256])
                if cv2.compareHist(maximum_hist,hist,3) > 0.04:
                    return False


    return True

def generate(
    seed=100,
    characters="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
    base_font="./LexendDeca-Medium.ttf",
    base_contents="Elden Ring, Final Fantasy, Granblue Fantasy, Atelier Ryza, and Harry Potter",
    prompt="sample of an useful vector bold font, in {base} style, designed with Inkscape, monochrome, no background",
    max_strength=0.9
):

    with autocast("cuda"):
        generator = torch.Generator("cuda")
        generator.manual_seed(seed)
        results = {}
        PROMPT = prompt.format(base=base_contents)
        i = 0
        for original_img,char,blocks,bounds,top_margin in generate_original_image(characters,base_font):
            init_img = preprocess(original_img)
            strength = max_strength+0
            while True:
                generated_image = (pipe(PROMPT,num_inference_steps=20,init_image=init_img,strength=strength,generator=generator)["sample"][0])
                extracted_image = extract_letters(np.asarray(generated_image),bounds,blocks)
                trimed_images = trim_letters(extracted_image)
                
                width_list = [p.shape[0] for p in trimed_images]
                height_list = [p.shape[0] for p in trimed_images]
                maximum_image = np.argmax(width_list+height_list)
                if match_letters(trimed_images):
                    results[char] = trimed_images[maximum_image]
                    break
                strength -= 0.01
                if strength < 0.5:
                    results[char] = trimed_images[maximum_image]
                    break
            i += 1
        i = 0
        for char,image in results.items():
            cv2.imwrite(f"./outputs/{str(i).zfill(len(str(len(characters))))}_{char}.png",image)
            i+=1
