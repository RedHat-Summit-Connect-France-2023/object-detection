import "@patternfly/react-core/dist/styles/base.css";
import React, { useState, useEffect, useCallback, } from "react";
import { connect } from "react-redux";

//import { Button } from "@mui/material";
import axios from "axios";
import { resetSearch, searchPhoto } from "../actions";
import {
  Form,
  FormGroup,
  TextInput,
  Checkbox,
  Popover,
  Divider,
  ActionGroup,
  Radio,
  Button,
  HelperText,
  HelperTextItem,
  FormHelperText,
  Bullseye,
  Card,
  CardHeader,
  CardActions,
  CardFooter,
  CardTitle,
  CardBody,
  Dropdown,
  DropdownToggle,
  DropdownItem,
  DropdownSeparator,
  DropdownPosition,
  DropdownToggleCheckbox,
  EmptyState,
  EmptyStateIcon,
  EmptyStateVariant,
  EmptyStateSecondaryActions,
  FormSelect,
  FormSelectOption,
  Gallery,
  KebabToggle,
  OverflowMenu,
  OverflowMenuControl,
  OverflowMenuDropdownItem,
  OverflowMenuItem,
  PageSection,
  PageSectionVariants,
  Pagination,
  Select,
  SelectOption,
  SelectVariant,
  TextContent,
  Text,
  Title,
  Toolbar,
  ToolbarItem,
  ToolbarFilter,
  ToolbarContent
} from '@patternfly/react-core';

import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {
  faCircleNotch,
  faSync,
  faExclamationCircle,
} from "@fortawesome/free-solid-svg-icons";
import { faCircle } from "@fortawesome/free-regular-svg-icons";
import QRCode from 'qrcode';

import "./Photo.scss";
import "./fonts.css"

const { INVENTORY_URL } = require("../../../utils/constants");

function Photo({
  reset,
  searchPhoto,
  predictionPending,
  predictionResponse,
  prediction,
  predictionError,
  minScore,
  labelSettings,
  status,
}) {

  const [image, setImage] = useState(null);
  const [catalog, setCatalog] = useState(false)
  const [productPrediction, setProductPrediction] = useState([])
  const [cameraEnabled, setCameraEnabled] = useState(null);
  const [clothes, setClothes] = useState([])
  const [video, setVideo] = useState(null);
  const [buttons,setButtons] = useState(false)
  const [videoWidth, setVideoWidth] = useState(0);
  const [videoHeight, setVideoHeight] = useState(0);
  const [imageCanvas, setImageCanvas] = useState(null);
  // eslint-disable-next-line no-unused-vars
  const [zonesCanvas, setZonesCanvas] = useState(null);
  const [facingMode, setFacingMode] = useState("environment");
  const [qrCodeUrl, setQRCodeUrl] = useState('');
  const [pseudo,setPseudo] = useState({pseudoName:"", pseudoId:-1})
  const [pseudoDefined,setPseudoDefined] = useState(false)
  const [shopWindow,setshopWindow] = useState(false)
  // InventoryImage is used to display an image from inventory when there is no prediction to display (display sellingItem)
  const [inventoryImage,setInventoryImage]= useState(false)
  const [itemToSell,setItemToSell] = useState([])
  const [itemToEdit,setItemToEdit] = useState([])
  const [pseudoUnicityError,setPseusoUnicityError] = useState(false)

useEffect(() => {
  // check if pseudo Defined
  console.log("USE EFFECT PSEUDO")
  const pseudo = JSON.parse(localStorage.getItem('pseudo'))
  setPseudoDefined((pseudo.pseudoId!==-1))
  setPseudo(pseudo)
}, []); 
  
  const userSession = 'aaabbbbaaaa'
  useEffect(() => {
    enableCamera();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    drawDetections();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [prediction]);

 useEffect(() => {
    const currentAddress = window.location.href;
    QRCode.toDataURL(currentAddress, function(err, url) {
      if (err) throw err;
      setQRCodeUrl(url);
    });
  }, []);
 
  const videoRef = useCallback(
    (node) => {
      setVideo(node);
      if (node) {
        navigator.mediaDevices
          .getUserMedia({ video: { facingMode } })
          .then((stream) => (node.srcObject = stream));
      }
    },
    [facingMode]
  );

  const imageCanvasRef = useCallback((node) => {
    setImageCanvas(node);
  }, []);

  const zonesCanvasRef = useCallback((node) => {
    setZonesCanvas(node);
  }, []);

  function enableCamera() {
    console.log("INVENTORY URL : ",INVENTORY_URL)

    setCameraEnabled(!cameraEnabled);
    setImage(null);
  }

  function onCameraToggled() {
    console.log("Take a new photo")
    reset();
    setshopWindow(false)
    setInventoryImage(false)
    setCatalog(false)
    enableCamera();
  }

  function onCameraClicked() {
    setQRCodeUrl(false)
    updateImageCanvas();

    let imageData = imageCanvas.toDataURL("image/jpeg");
    const base64data = imageData.replace(/^data:image\/(png|jpg|jpeg);base64,/, "");
    searchPhoto(base64data);
  }

  function updateImageCanvas() {
    setVideoWidth(video.videoWidth);
    setVideoHeight(video.videoHeight);
    if (!imageCanvas) {
      return;
    }

    imageCanvas.width = video.videoWidth;
    imageCanvas.height = video.videoHeight;

    imageCanvas.getContext("2d").drawImage(video, 0, 0, video.videoWidth, video.videoHeight);

    video.srcObject.getVideoTracks().forEach((track) => {
      track.stop();
    });
    setImage(imageCanvas.toDataURL());

    setCameraEnabled(false);
  }

  function drawDetections() {
    if (!prediction || !prediction.detections || !imageCanvas.getContext) {
      return;
    }
    const displayBox = prediction.displayBox;

    console.log(" PREDICITION ",prediction['detections'])
    console.log(" PREDICITION . ",prediction.detections)
    console.log(" TO CLOTHEOBJECT PREDICTION ",prediction)
    // empty ItemToSell before a new prediction 
    setItemToSell([])
    toClotheObject(prediction['detections'],prediction.image_url)
    //display item to edit
    setshopWindow(true)
    setButtons(true)
    prediction.detections.filter((d) => d.score > minScore).forEach((d) => drawDetection(d,displayBox));
  }

  function drawDetection({ box, label, score, cValue }, displayBox) {
    const drawScore = true;
    const textBgHeight = 14;
    const padding = 2;
    const letterWidth = 6;
    const scoreWidth = drawScore ? 4 * letterWidth : 0;
    const text = drawScore ? `${label} ${Math.floor(score * 100)}% Confidence` : label;

    const width = Math.floor((box.xMax - box.xMin) * imageCanvas.width);
    const height = Math.floor((box.yMax - box.yMin) * imageCanvas.height);
    const x = Math.floor(box.xMin * imageCanvas.width);
    const y = Math.floor(box.yMin * imageCanvas.height);
    const labelSetting = labelSettings[label];
    const labelWidth = text.length * letterWidth + scoreWidth + padding * 2;

    const ctx = imageCanvas.getContext("2d");
    if (displayBox) {
      drawBox(ctx, x, y, width, height, labelSetting.bgColor);
      drawBoxTextBG(ctx, x, y + height - textBgHeight, labelWidth, textBgHeight, labelSetting.bgColor);
      drawBoxText(ctx, text, x + padding, y + height - padding);
    }
    //clearZone(ctx, x + 5, y + height - textBgHeight - 4, labelWidth, textBgHeight);
    //clearZone(ctx, x, y, width, height);
  }

  function drawBox(ctx, x, y, width, height, color) {
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 15]);
    ctx.strokeStyle = color;
    ctx.strokeRect(x, y, width, height);
  }

  function drawBoxTextBG(ctx, x, y, width, height, color) {
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.5;
    ctx.fillRect(x, y, width, height);
    ctx.globalAlpha = 1.0;
  }

  function drawBoxText(ctx, text, x, y) {
    ctx.font = "12px Verdana";
    ctx.fillStyle = "white";
    ctx.fillText(text, x, y);
  }

  function onFacingModeClicked() {
    if (facingMode === "user") {
      setFacingMode("environment");
    } else {
      setFacingMode("user");
    }
  }




  function renderCamera() {
    
    if (!cameraEnabled || image) {

      return null;
    }

    const displayResult = pseudoDefined ? {} : { display: "none" };

    //after prediction returned
    // display the sellingLayout
    return (

      <div className="camera" style={displayResult}>
        <div className="img-preview">
          <div className="img-container">
            <video
              className="camera-preview"
              ref={videoRef}
              controls={false}
              autoPlay
              playsInline
            />
          </div>
        </div>
        <div className="left-button-container button-container">
          <Button
            variant="contained"
            size="large"
            className="choose-camera-button"
            onClick={onFacingModeClicked}
          >
            <FontAwesomeIcon icon={faSync} />
          </Button>
        </div>
        <div className="center-button-container button-container">
          <Button
            variant="contained"
            size="large"
            className="take-picture-button"
            onClick={onCameraClicked}
          >
            <FontAwesomeIcon icon={faCircle} />
          </Button>
        </div>
      </div>
    );
  }

  function renderSnapshot() {
    const displayResult = image ? {} : { display: "none" };


    const displayButtons = predictionPending ? { display: "none" } : {};
    const displayLoading = predictionPending ? {} : { display: "none" };

    //      (!predictionPending && predictionError) || (!inventoryImage)

   const displayError =
      (!predictionPending && predictionError)
        ? { width: `${videoWidth}px`, height: `${videoHeight}px` }
        : { display: "none" };

    const displayImage =
      (!predictionPending && !predictionError && prediction) ? {} : { display: "none" };

    let displayNoObjects;

    // Get Predictions from IA inference
    console.log(" DISPLAY LOADING ? ",displayLoading)
    console.log(" DISPLAY RESULT ? ",displayResult)
    console.log(" DISPLAY IMAGE ? ",displayImage)
    console.log(" DISPLAY ERROR ? ",displayError)
    

    displayNoObjects = { display: "none" }; // Never show no objects
    return (

      <div className="result" style={displayResult}>
        <div className="img-preview">
          <div className="error-container" style={displayError}>
            <h2>
              <FontAwesomeIcon className="error-icon" icon={faExclamationCircle} /> Error
            </h2>
            <code>{JSON.stringify(predictionError, null, 2)}</code>
          </div>
          <div className="img-container" style={displayImage}>
            <canvas className="result-canvas" ref={imageCanvasRef} />
            <div className="zones overlay">
              <canvas className="zones-canvas" ref={zonesCanvasRef} />
            </div>
            <div className="loading overlay" style={displayLoading}>
              <div>
                <FontAwesomeIcon className="loading-icon" icon={faCircleNotch} spin />
              </div>
              <div className="loading-text">Loading ...</div>
            </div>
            <div className="no-objects overlay" style={displayNoObjects}>
              <div className="no-objects-text">No Objects</div>
              <div className="no-objects-text">Found</div>
            </div>
          </div>
        </div>
        <div className="right-button-container button-container" style={displayButtons}></div>
      </div>
    );
  }

  function renderQRCode() {
   // const displayQRcode = image ? {} : { display: "none" };
   // hide qr code
   // to do , add main variable outside as image ..

   const displayQRcode = !pseudoDefined ? {} : { display: "none" };
    return (
      <div className="img-preview">
        <img src={qrCodeUrl} alt="QR Code" style={displayQRcode} />
      </div>
    );
  }

  function renderShopwindowPhotoButton(){

    const displayButtons = buttons ? {} : { display: "none" };

      return (
    <div className="shop-button" style={displayButtons}>
    <Button
        variant="contained"
        size="sm"
        onClick={getCatalog}
      >
        <span className="label-word">Show my dressing</span>
      </Button>{' '}
        <Button
         variant="contained"
         size="sm"
    
        onClick={onCameraToggled}
      >
        <span className="label-word">Take a new picture ! </span>
      </Button>
    </div>
)

  }

  const handlePseudoCreation = (_event) => {

    
      axios.post(INVENTORY_URL+'/pseudos', 
      {"pseudoName" : pseudo.pseudoName}
    ,{headers:
      {'Content-Type': 'application/json'}
        })
    .then((response) => {
      var idCreated = response.data
      pseudo.pseudoId= idCreated

      setPseudo(pseudo)
      console.log("Id  created ",idCreated);

      console.log("NEW Pseudo ",pseudo)
      localStorage.setItem('pseudo', JSON.stringify(pseudo));
      setQRCodeUrl(false)
      setPseusoUnicityError(false)
      setPseudoDefined(true)
      setButtons(true)
    
    }, (error) => {
      console.log("POST ",error);
      setPseusoUnicityError(true)
    });
   


  }

  const handlePseudoName = (_event,name) => {
    
    setPseudo({...pseudo,pseudoName:name})
    console.log("Pseudo change ",pseudo)
  }

  function definePseudo(){
    
    console.log("PSEUDO LOCAL ", JSON.parse(localStorage.getItem('pseudo')))
 if (JSON.parse(localStorage.getItem('pseudo')== null)){
  console.log(" PSEUDO LOCAL 2",pseudo)
      localStorage.setItem('pseudo', JSON.stringify(pseudo));

    }
   
    const displayPseudo =  !pseudoDefined ? {}: { display: "none" };
    const displayPseudoError = pseudoUnicityError ? {} : { display: "none" };
    console.log(" PSEUDO Displayed ",displayPseudo)
    console.log ( "PSEUDO Defined ",pseudoDefined)

    return (
      <Card ouiaId="BasicCard" style={displayPseudo}>
    <CardTitle>Red Hat Peer-to-Peer Shop</CardTitle>
    <CardBody>
  
      <Form>
            <FormGroup
              label="Saisissez un pseudo pour votre dressing"
                  
              isRequired
              fieldId="simple-form-name-01"
            >
              <TextInput
                isRequired
                type="text"
                id="simple-form-name-01"
                name="userPseudo"
                aria-describedby="simple-form-name-01-helper"
                onChange={handlePseudoName}
              />
               <HelperText style={displayPseudoError}>
      <HelperTextItem variant="error">Your pseudo is already taken !</HelperTextItem>
             </HelperText>

              </FormGroup>
            <ActionGroup>
              <Button variant="primary" onClick={handlePseudoCreation}>C'est parti !</Button>
            </ActionGroup>
            </Form>
            </CardBody>
            </Card>
    )
  }

  const handleSellItem = (e,index,attribute) => {

    
    const updatedObject = itemToSell[index]
    setItemToEdit({...itemToEdit,[attribute]:e})

    updatedObject[attribute]= e

    console.log("Updated Object",updatedObject)
  console.log("clik item", "value ", e," index ",index," attribute",attribute)


  };


  const handleEditItem = (index) => {
    return() =>{
    
    // draw image from url in canvas
   

    const dataset1 = {
      detections: [
      {
      box: [Object],
      class: 'haut à manches courtes',
      label: 'Haut à manches courtes',
      score: 0.6621403098106384
      }
      ],
      image_url: 'https://www.teez.in/cdn/shop/products/Red-Hat-T-Shirt-2_fa4655dc-8606-4d52-be50-787f19bd4310_large.jpg?v=1580794873'
      }

      setInventoryImage(clothes[index].link)
  

console.log("Clothes To Edit ", clothes)
  

   
    //setProductPrediction(dataset1["detections"])
    //setItemToSell(dataset1["detections"])

    
  

    // clothes set to comment for testing object detection
    //setClothes([clothes[index]])
    toClotheObject([clothes[index]])
    setButtons(true)
    setImage(null)
    setshopWindow(true)
    setCatalog(false)
 
 
  }
}

function renderInventoryImage(){

  const displayInventoryImage= inventoryImage ? {} : { display: "none" };

  return(
    <div className="inventoryImage"  style={displayInventoryImage}> 
      <img alt='' className="inventoryImageCanva" src={inventoryImage} style={{height:imageCanvas?.height, width:imageCanvas?.width}} />

    </div>

  )
}

const toClotheObject = (products,predicted_image_url) =>{

  console.log("To Clothe Object ",products)
  console.log("To Clothe Object IMG  ",predicted_image_url)
  products.map(function(o,i) {
 
    const clothObj = {
    "description": "",
        "itemId": null,
        "link": "",
        "location": "",
        "name": "",
        "price": 0.0,
        "quantity": 1,
        "pseudoId":"",
        "category":"",
        "numero":null,
        "confident":null
    }

   /*  const clothObj =
    {
      id: "",
      name: "",
      class: "",
      description: "",
      price: '',
      image_url:""
    } */
    // set price to 0 when it is predicted
    if (o.score!= null){
      clothObj.confident=Math.floor(o.score * 100)
    }

    if (o.price!= null){
      clothObj.price=o.price
    }
    if (o.link==null){
      clothObj.link= predicted_image_url
    }else{
      clothObj.link= o.link
    }
    if (o.itemId!=null){
      clothObj.itemId = o.itemId
    }

    if (o.class!=null){
      clothObj.category= o.class
    }
    else{
      clothObj.category= o.category
    }
    
    clothObj.pseudoId = pseudo.pseudoId
    clothObj.numero = i
    clothObj.name= o.name
    clothObj.description=o.description

  
      return(
      setItemToSell((prevItems)=> [...prevItems,clothObj]) )
  })
}






const  removeItemEdited = useCallback((index) => setItemToSell((current) =>
current.filter((itemToSell) => itemToSell.numero !== index)
), []);




const sendToInventory = (index) => {
  return() =>{

  console.log("INDEX INVENTORY ",index)

  console.log(" NEW ITEM TO SELL",itemToSell)
    if (itemToSell.length === 1){
      console.log("All items predicted sold" )
    
      setImage(null)
      setInventoryImage("https://www.sifisa.fr/images/0093900001574926432.jpg") // hide detected image with predictions
      setCatalog(true)
      setshopWindow(false) 
  //console.log(" RAW INVENTORY ", itemToSell)
    }
  let payload = itemToSell[index]
  // for test purpose , image is forced
 // payload.link= "https://www.teez.in/cdn/shop/products/Red-Hat-T-Shirt-2_fa4655dc-8606-4d52-be50-787f19bd4310_large.jpg?v=1580794873"
  delete payload["index"];
  delete payload["confident"]
  console.log("SEND TO INVENTORY ",payload)

  axios.post(INVENTORY_URL+'/products', 
    payload
  ,{headers:
    {'Content-Type': 'application/json'}
      })
  .then((response) => {
    removeItemEdited(index)
  
      getCatalog()
    console.log("POST 2 ",response);
  }, (error) => {
    console.log("POST ",error);
  });

  
}
}


  



 function renderShopWindow(){
  
        const shopWindowDisplay = shopWindow ? {} : { display: "none" };
        const options = [
          { value: "haut à manches courtes" , label : "haut à manches courtes" },
          { value: "haut à manches longues" , label : "haut à manches longues" },
          { value: "veste à manches courtes" , label : "veste à manches courtes" },
          { value: "veste à manches longues" , label : "veste à manches longues" },
          { value: "gillet" , label : "gillet" },
          { value: "écharpe" , label : "écharpe" },
          { value: "short" , label : "short" },
          { value: "pantalon" , label : "pantalon" },
          { value: "jupe" , label : "jupe" },
          { value: "robe à manche courtes" , label : "robe à manche courtes" },
          { value: "robe à manches longues" , label : "robe à manches longues" },
          { value: "robe" , label : "robe" },
        ]
       
        const quantities = [
          { value: "0" , label : "0" },
          { value: "1" , label : "1" },
          { value: "2" , label : "2" },
          { value: "3" , label : "3" },
          { value: "4" , label : "4" },
          { value: "5" , label : "5" },
      

        ]
        const prices = [
          { value: "0" , label : "Free" },
          { value: "1" , label : "1" },
          { value: "2" , label : "2" },
          { value: "5" , label : "5" },
          { value: "10" , label : "10" },
          { value: "15" , label : "15" },
          { value: "20" , label : "20" }
        ]

        console.log("ITEM TO SELL ",itemToSell)

    
        
     
        return(
          <div className="sell-item"  style={shopWindowDisplay}> 
    <div class="pf-v5-l-gallery pf-m-gutter">

          {itemToSell.map((clothe,index) =>(
             

          <Card ouiaId="BasicCard">
          <CardBody>
  
        <Form>
        <FormGroup  label={"AI model predicted this category : " + clothe.category +" with "+clothe.confident +"% confidence. Change it if not accurate"}>
        <FormSelect 
          id="category"
          aria-label="FormSelect Input"
          value={clothe.category}
          onChange={e => handleSellItem(e.target.value,index,"category")}
          isRequired fieldId="simple-form-name-03" >
          {options.map((option, index) => (
            <FormSelectOption
              key={index}
              value={option.value}
              label={option.label}
            />
          ))}
        </FormSelect>
        </FormGroup>
        <FormGroup
          label="Name" fieldId="simple-form-name-01" >
          <TextInput type="text" onChange={e => handleSellItem(e.target.value,index,"name")} value={clothe.name} id="simple-form-name-02" name="itemName" aria-describedby="simple-form-name-01-helper"/>
        </FormGroup>
        <FormGroup
          label="Color" fieldId="simple-form-name-02" >
          <TextInput type="text" onChange={e => handleSellItem(e.target.value,index,"description")} value={clothe.description} id="simple-form-name-03" name="descriptipn" aria-describedby="simple-form-name-01-helper"/>
        </FormGroup>
        <FormGroup label="Quantity to sell">
           <FormSelect 
           onChange={e => handleSellItem(e.target.value,index,"quantity")}
           id="price"
           aria-label="FormSelect Input"
           value={clothe.quantity}
           isRequired fieldId="simple-form-name-04" >
           {quantities.map((qty, index) => (
             <FormSelectOption
               key={index}
               value={qty.value}
               label={qty.label}
             />
           ))}
         </FormSelect>
         </FormGroup>
        <FormGroup label="Price (€)">
           <FormSelect 
           onChange={e => handleSellItem(e.target.value,index,"price")}
           id="price"
           aria-label="FormSelect Input"
           value={clothe.price}
           isRequired fieldId="simple-form-name-04" >
           {prices.map((price, index) => (
             <FormSelectOption
               key={index}
               value={price.value}
               label={price.label}
             />
           ))}
         </FormSelect>
         </FormGroup>
        <ActionGroup>
          <Button variant="primary" onClick={sendToInventory(index)} >Sell !</Button>
        </ActionGroup>
        </Form>
        </CardBody>
        </Card>
          ))
          }
    </div>
          </div>
        )
      }
  
  function renderCatalog(){
    const productsToSellLayout = catalog ? {} : { display: "none" };
    return (
    <div className="item-catalog" style={productsToSellLayout}>
        <div
      class="pf-v5-l-gallery pf-m-gutter"
    >
  
  {clothes.map((clothe,index) =>(

  
  
  <Card ouiaId="BasicCard">

    <CardTitle>{clothe.category} - {clothe.name}</CardTitle>
    <CardTitle>{clothe.description}</CardTitle>
    <CardBody>Quantity remaining : <div class="pf-u-font-weight-bold">{clothe.quantity}</div></CardBody>  
    <CardBody>{clothe.price} €</CardBody>
    <CardFooter>
    <Button onClick={handleEditItem(index)} variant="secondary" size="small">
      Edit
    </Button>
    </CardFooter>
    <Divider component="div" />

  </Card>


))}
</div>
</div>)

    
  }



  
/*
a ajouter au apres snapshot
 {renderInventoryImage()}

 */

         
         

  
  return (
    <div className="photo">
      <div>
    <Title class="pf-v5-u-text-align-center" headingLevel="h1"> The Good Corner </Title>
    <Title class="pf-v5-u-text-align-center" headingLevel="h2"> Welcome {pseudo.pseudoName} </Title>

    <br></br>
    </div>

    {definePseudo()}
    {renderQRCode()}
    {renderCamera()}
    {renderSnapshot()}
    {renderInventoryImage()}
    {renderShopwindowPhotoButton()}
    {renderCatalog()}
    {renderShopWindow()}
      
    </div>
  );

 
  
  



  function getCatalog(){

    // route to update to get items by userId
   // https://inventory-thegoodcorner-dev.apps.summitconnect.sandbox2218.opentlc.com/products/pseudo.pseudoId
  console.log("INVENTORY URL : ",INVENTORY_URL)
    axios.get(INVENTORY_URL+'/products/pseudo/'+ pseudo.pseudoId)
   .then(response => {

    let items = response?.data;

    console.log(" CATALOG ITEM ",items)

     setClothes(items)
     setshopWindow(false)
     setItemToSell([])
     setImage(null)
     setCameraEnabled(false)
     setInventoryImage("https://www.sifisa.fr/images/0093900001574926432.jpg")
     setCatalog(true)
     

    //console.log("photo prediction ",prediction.detections)
    console.log("get catalog",response)

  })
  .catch(error=>{
    console.error("error api",error)
  })

}
}



function mapStateToProps(state) {
  return { ...state.appReducer, ...state.photoReducer };
}

function mapDispatchToProps(dispatch) {
  return {
    reset: () => {
      dispatch(resetSearch());
    },
    searchPhoto: (photo) => {
      dispatch(searchPhoto(photo));
    },
  };
}

export default connect(mapStateToProps, mapDispatchToProps)(Photo);
