# --- Stage 1: Build ---
FROM node:20-slim as build

WORKDIR /app

# Copy package files and install dependencies
COPY web_ui/package*.json ./
RUN npm install

# Copy source and build
COPY web_ui/ ./
RUN npm run build

# --- Stage 2: Serve ---
FROM nginx:stable-alpine

# Copy build artifacts from previous stage
COPY --from=build /app/dist /usr/share/nginx/html

# Expose port (Nginx default)
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
