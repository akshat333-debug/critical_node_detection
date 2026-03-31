# Use an official Node runtime as a parent image
FROM node:20-slim AS build

# Set the working directory in the container
WORKDIR /app

# Copy the package.json and package-lock.json into the container at /app
COPY package.json package-lock.json ./

# Install any needed packages specified in package.json
RUN npm install

# Copy the rest of the frontend code into the container at /app
COPY . .

# Expose port 5173
EXPOSE 5173

# Start Vite dev server with --host to enable external access
CMD ["npm", "run", "dev", "--", "--host"]
