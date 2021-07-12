import React, {useState,useEffect} from "react";
import { Link } from "react-router-dom";
import { Row, Col, Image, ListGroup, Card, Button } from "react-bootstrap";
import axios from 'axios'
// import products from "../products";

const ProductScreen = ({ match }) => {
  // const product = products.find((p) => p._id === match.params.id);

  const [product,setProduct] = useState({})

  useEffect(() => {
    const fetchProduct = async () => {
        const {data} = await axios.get(`/api/products/${match.params.id}`)

        setProduct(data)
    }

    fetchProduct()
},[])

  return (
    <>
      <Link className="btn btn-dark my-3" to="/products">
        Go Back
      </Link>
      <Row>
        <Col md={6}>
          <Card className=" p-3 rounded">
            <Image src={product?.image} variant="top" fluid />
          </Card>
        </Col>
        <Col md={3}>
          <ListGroup variant="flush">
            <ListGroup.Item>
              <h4>{product?.name}</h4>
            </ListGroup.Item>
            <ListGroup.Item>{product?.description}</ListGroup.Item>
            <ListGroup.Item><Row><Col>Weight :</Col><Col>{product?.weight}</Col></Row></ListGroup.Item>
            <ListGroup.Item><Row><Col>Height :</Col><Col>{product?.height}</Col></Row></ListGroup.Item>
            <ListGroup.Item><Row><Col>Life-Span :</Col><Col>{product?.lifeSpan}</Col></Row></ListGroup.Item>
       
          </ListGroup>
        </Col>
      </Row>
      <Row>
        {product?.dogProd?.map((prod) => (
          <Col md={3}>
            <Card className="my-3 p-3 rounded">
              <Card.Img
                src={prod?.img}
                variant="top"
                style={{
                  objectFit: "contain",
                  width: "20vw",
                  height: "30vh",
                }}
              />
              <Card.Body>
                <Card.Title as="div">
                  <strong>{prod?.name}</strong>
                </Card.Title>
              </Card.Body>
            </Card>
          </Col>
        ))}
      </Row>
    </>
  );
};

export default ProductScreen;
