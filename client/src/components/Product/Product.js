import React from "react";
import { Card } from "react-bootstrap";
import {Link} from 'react-router-dom'
const Product = ({ product }) => {
  return (
    <Card className="my-3 p-3 rounded">
      <Link to={`/products/${product._id}`}>
        <Card.Img
          src={product.image}
          variant="top"
          style={{
            objectFit: "contain",
            width: "20vw",
            height: "30vh",
          }}
        />
      </Link>
      <Card.Body>
      <Link to={`/products/${product._id}`}>
       <Card.Title as='div' >
          <strong>{product.name}</strong>
       </Card.Title>
      </Link>
      {/* <Card.Text as='div'></Card.Text> */}
      </Card.Body>
    </Card>
  );
};

export default Product;
